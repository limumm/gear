#!/usr/bin/env python3
"""
SAM Mask Generation Script for ArtGS Dataset

This script processes images from the ArtGS dataset using SAM (Segment Anything Model)
to generate masks under data/<dataset_path>/.../train/mask (*.npy). By default it does
not write visualization outputs; pass --save_visualization to also save overlays under outputs/.

Usage:
    python get_mask.py --dataset_path artgs/sapien/storage_47648
    python get_mask.py --dataset_path artgs/sapien/storage_47648 --save_visualization
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import SAM modules
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class MaskProcessor:
    """Main class for processing masks using SAM"""
    
    def __init__(self, dataset_path: str, sam_checkpoint: str = "submodules/sam_vit_h_4b8939.pth", 
                 alpha_threshold: float = 0.5, save_visualization: bool = False):
        """
        Initialize the mask processor
        
        Args:
            dataset_path: Path to the dataset (e.g., "artgs/sapien/storage_47648")
            sam_checkpoint: Path to SAM checkpoint file
            alpha_threshold: Alpha channel threshold for foreground detection (0.0-1.0)
            save_visualization: If True, also save PNG visualizations under outputs/
        """
        self.dataset_path = dataset_path
        self.sam_checkpoint = sam_checkpoint
        self.alpha_threshold = alpha_threshold
        self.save_visualization = save_visualization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup paths
        self.data_path = os.path.join("data", dataset_path)
        self.outputs_path = os.path.join("outputs", dataset_path)
        
        # Initialize SAM model
        self._initialize_sam()
        
    def _initialize_sam(self):
        """Initialize SAM model and mask generator"""
        print(f"Initializing SAM model on {self.device}...")
        
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        
        # Configure mask generator with parameters from the notebook
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        print("SAM model initialized successfully!")
    
    def load_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess RGBA image for SAM
        
        Returns:
            Tuple of (rgb_image, alpha_mask) where:
            - rgb_image: RGB image for SAM processing
            - alpha_mask: Binary mask indicating foreground regions
        """
        # Load RGBA image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Check if image has alpha channel
        if image.shape[2] == 4:
            # Extract RGB and alpha channels
            rgb_image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
            alpha_channel = image[:, :, 3]
            
            # Create binary mask for foreground (alpha > threshold)
            alpha_mask = (alpha_channel / 255.0) > self.alpha_threshold
            alpha_mask = alpha_mask.astype(np.uint8) * 255
            
        else:
            # If no alpha channel, treat entire image as foreground
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            alpha_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        
        return rgb_image, alpha_mask
    
    def generate_sam_masks(self, image: np.ndarray, alpha_mask: np.ndarray = None) -> List[Dict]:
        """
        Generate masks using SAM, optionally filtered by alpha mask
        
        Args:
            image: RGB image for SAM processing
            alpha_mask: Binary mask indicating foreground regions (optional)
        
        Returns:
            List of SAM mask dictionaries, filtered to only include masks in foreground regions
        """
        # Generate all masks using SAM
        masks = self.mask_generator.generate(image)
        
        # If alpha mask is provided, filter masks to only include those in foreground regions
        if alpha_mask is not None:
            filtered_masks = []
            for mask_dict in masks:
                mask = mask_dict['segmentation']
                
                # Calculate overlap between SAM mask and alpha mask
                overlap = np.logical_and(mask, alpha_mask > 0)
                overlap_ratio = np.sum(overlap) / np.sum(mask) if np.sum(mask) > 0 else 0
                
                # Only keep masks that have significant overlap with foreground
                if overlap_ratio > 0.3:  # At least 30% of the mask should be in foreground
                    # Update the mask to only include foreground regions
                    mask_dict['segmentation'] = np.logical_and(mask, alpha_mask > 0)
                    mask_dict['area'] = np.sum(mask_dict['segmentation'])
                    
                    # Only keep masks with sufficient area after filtering
                    if mask_dict['area'] > 50:  # Minimum area threshold
                        filtered_masks.append(mask_dict)
            
            return filtered_masks
        
        return masks
    
    def show_anns(self, anns, ax=None):
        """Display SAM masks with random colors (from notebook)"""
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        if ax is None:
            ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
    
    def save_sam_masks(self, sam_masks: List[Dict], output_dir: str, 
                      image_name: str):
        """
        Save SAM masks to output directory (npy format only)
        
        Args:
            sam_masks: List of SAM mask dictionaries
            output_dir: Output directory path
            image_name: Base name of the image (e.g., "0000")
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Sort masks by area (largest first) and select top masks
        sorted_masks = sorted(sam_masks, key=lambda x: x['area'], reverse=True)
        
        # Save combined mask array as npy file only
        if sorted_masks:
            combined_mask = self._create_combined_mask(sorted_masks)
            npy_path = os.path.join(output_dir, f"{image_name}.npy")
            np.save(npy_path, combined_mask)
    
    def _create_combined_mask(self, sam_masks: List[Dict]) -> np.ndarray:
        """
        Create a combined mask array where each mask gets a unique value
        
        Args:
            sam_masks: List of SAM mask dictionaries
            
        Returns:
            Combined mask array with shape (height, width) where each pixel
            contains the mask ID (1, 2, 3, ...) or 0 for background
        """
        if not sam_masks:
            return np.array([])
        
        # Get image dimensions from the first mask
        first_mask = sam_masks[0]['segmentation']
        height, width = first_mask.shape
        combined_mask = np.zeros((height, width), dtype=np.uint16)
        
        # Assign unique values to each mask (starting from 1)
        for i, mask_dict in enumerate(sam_masks, 1):
            mask = mask_dict['segmentation']
            # Set pixels belonging to this mask to the mask ID
            combined_mask[mask] = i
        
        return combined_mask
    
    def save_visualization(self, image: np.ndarray, sam_masks: List[Dict], 
                          output_dir: str, image_name: str, alpha_mask: np.ndarray = None):
        """
        Save visualization of SAM masks overlaid on original image
        
        Args:
            image: Original RGB image
            sam_masks: List of SAM mask dictionaries
            output_dir: Output directory path
            image_name: Base name of the image (e.g., "0000")
            alpha_mask: Alpha mask for foreground regions (optional)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualization with alpha mask overlay
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title(f'Original Image: {image_name}')
        axes[0].axis('off')
        
        # Alpha mask visualization
        if alpha_mask is not None:
            axes[1].imshow(alpha_mask, cmap='gray')
            axes[1].set_title(f'Alpha Mask (threshold={self.alpha_threshold})')
            axes[1].axis('off')
        else:
            axes[1].text(0.5, 0.5, 'No Alpha Mask', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Alpha Mask')
            axes[1].axis('off')
        
        # SAM masks overlaid on original image
        axes[2].imshow(image)
        self.show_anns(sam_masks, ax=axes[2])
        axes[2].set_title(f'SAM Masks ({len(sam_masks)} masks)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        vis_filename = f"{image_name}_sam_visualization.png"
        vis_path = os.path.join(output_dir, vis_filename)
        plt.savefig(vis_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        # Also save alpha mask separately
        if alpha_mask is not None:
            alpha_filename = f"{image_name}_alpha_mask.png"
            alpha_path = os.path.join(output_dir, alpha_filename)
            cv2.imwrite(alpha_path, alpha_mask)
    
    @staticmethod
    def load_sam_masks(mask_dir: str, image_name: str) -> List[Dict]:
        """
        Load SAM masks from saved file (deprecated - pkl files no longer saved)
        
        Args:
            mask_dir: Directory containing the mask files
            image_name: Base name of the image (e.g., "0000")
            
        Returns:
            List of SAM mask dictionaries (empty list since pkl files no longer saved)
        """
        print(f"Warning: load_sam_masks is deprecated - pkl files are no longer saved")
        return []
    
    @staticmethod
    def load_combined_mask(mask_dir: str, image_name: str) -> np.ndarray:
        """
        Load combined mask array from saved npy file
        
        Args:
            mask_dir: Directory containing the mask files
            image_name: Base name of the image (e.g., "0000")
            
        Returns:
            Combined mask array with shape (height, width) where each pixel
            contains the mask ID (1, 2, 3, ...) or 0 for background
        """
        npy_path = os.path.join(mask_dir, f"{image_name}.npy")
        if os.path.exists(npy_path):
            combined_mask = np.load(npy_path)
            return combined_mask
        else:
            print(f"Warning: No combined mask found for {image_name}")
            return np.array([])
    
    @staticmethod
    def load_all_masks_from_sequence(mask_dir: str) -> Dict[str, List[Dict]]:
        """
        Load all masks from a sequence directory (deprecated - pkl files no longer saved)
        
        Args:
            mask_dir: Directory containing mask files for a sequence
            
        Returns:
            Dictionary mapping image names to sam_masks lists (empty since pkl files no longer saved)
        """
        print(f"Warning: load_all_masks_from_sequence is deprecated - pkl files are no longer saved")
        return {}
    
    def process_sequence(self, sequence_type: str, start_from: int = 0):
        """
        Process a sequence of images (start or end)
        
        Args:
            sequence_type: Either "start" or "end"
            start_from: Image number to start from (e.g., 25 to start from 0025.png)
        """
        # Define paths
        rgba_path = os.path.join(self.data_path, sequence_type, "train", "rgba")
        mask_path = os.path.join(self.data_path, sequence_type, "train", "mask")
        vis_path = None
        if self.save_visualization:
            vis_path = os.path.join(self.outputs_path, f"sam_vis{0 if sequence_type == 'start' else 1}")
        
        # Check if paths exist
        if not os.path.exists(rgba_path):
            print(f"Warning: RGBA path does not exist: {rgba_path}")
            return
        
        # Get list of images
        image_files = sorted([f for f in os.listdir(rgba_path) if f.endswith('.png')])
        
        # Filter images based on start_from parameter
        if start_from > 0:
            start_filename = f"{start_from:04d}.png"
            try:
                start_index = image_files.index(start_filename)
                image_files = image_files[start_index:]
            except ValueError:
                tqdm.write(f"Warning: Image {start_filename} not found in {sequence_type}, starting from beginning.")
        
        desc = f"{sequence_type} ({len(image_files)} img)"
        for image_file in tqdm(image_files, desc=desc, unit="img", leave=True):
            image_name = os.path.splitext(image_file)[0]  # e.g., "0000"

            # Load image and alpha mask
            image_path = os.path.join(rgba_path, image_file)
            try:
                image, alpha_mask = self.load_image(image_path)
            except Exception as e:
                tqdm.write(f"Error loading {image_file}: {e}")
                continue
            
            # Generate SAM masks with alpha filtering
            try:
                sam_masks = self.generate_sam_masks(image, alpha_mask)
            except Exception as e:
                tqdm.write(f"Error SAM {image_file}: {e}")
                continue
            
            # Save individual masks
            if sam_masks:
                self.save_sam_masks(sam_masks, mask_path, image_name)
                if self.save_visualization and vis_path is not None:
                    self.save_visualization(image, sam_masks, vis_path, image_name, alpha_mask)
            else:
                tqdm.write(f"No SAM masks: {image_file}")
    
    def process_dataset(self, start_from: int = 0):
        """Process the entire dataset"""
        extra = f", end from {start_from:04d}" if start_from > 0 else ""
        print(f"Dataset {self.dataset_path}{extra}  |  data: {self.data_path}")
        self.process_sequence("start", 0)
        self.process_sequence("end", start_from)
        print(f"Done: {self.dataset_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate SAM masks for ArtGS dataset")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset (e.g., 'artgs/sapien/storage_47648')")
    parser.add_argument("--sam_checkpoint", type=str, default="submodules/sam_vit_h_4b8939.pth",
                       help="Path to SAM checkpoint file")
    # parser.add_argument("--max_masks", type=int, default=10,
    #                    help="Maximum number of masks to save per image")
    parser.add_argument("--start_from", type=int, default=0,
                       help="Image number to start from (e.g., 25 to start from 0025.png)")
    parser.add_argument("--alpha_threshold", type=float, default=0.5,
                       help="Alpha channel threshold for foreground detection (0.0-1.0)")
    parser.add_argument("--save_visualization", action="store_true",
                       help="Also save SAM overlay and alpha PNGs under outputs/<dataset_path>/sam_vis*")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.sam_checkpoint):
        print(f"Error: SAM checkpoint not found at {args.sam_checkpoint}")
        return
    
    # Create processor and run
    try:
        processor = MaskProcessor(
            args.dataset_path,
            args.sam_checkpoint,
            args.alpha_threshold,
            save_visualization=args.save_visualization,
        )
        processor.process_dataset(args.start_from)
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()