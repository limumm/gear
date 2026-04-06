#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import json
import os
import sys
from random import randint

import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm
from argparse import ArgumentParser
from pytorch_lightning import seed_everything
from pytorch3d.ops import knn_points
from sklearn.neighbors import KDTree

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import DeformModel, GaussianModel, Scene
from utils.dual_quaternion import matrix_to_quaternion
from utils.general_utils import get_linear_noise_func, safe_state, vis_depth
from utils.log_utils import prepare_output_and_logger
from utils.loss_utils import l1_loss, ssim
from utils.metrics import *


class Trainer:
    def __init__(self, args, dataset, opt, pipe, saving_iterations, coarse_name="coarse_point_cloud", coarse_iter=0):
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.saving_iterations = saving_iterations

        self.tb_writer = prepare_output_and_logger(args)
        
        self.gaussians = GaussianModel(dataset.sh_degree, num_joints=args.num_slots)
        self.gaussians1 = GaussianModel(dataset.sh_degree, num_joints=args.num_slots)

        self.dataset.iterations = args.iterations
        self.dataset.coarse_name = coarse_name
        self.dataset.coarse_iter = coarse_iter
        self.dataset.model_path_noseed = args.model_path_noseed
        self.deform = DeformModel(self.dataset)
        print('Init GaussianModel and DeformModel.')
        print(f"Coarse name: {coarse_name}")
        self.larger_motion = open(os.path.join(args.model_path_noseed, coarse_name,"point_cloud", f"iteration_{coarse_iter}", "larger_motion_state.txt"), "r").read()
        if self.larger_motion == "1":
            larger_motion_path = os.path.join(args.model_path_noseed, coarse_name, "point_cloud", f"iteration_{coarse_iter}", "point_cloud_1.ply")
        else:
            larger_motion_path = os.path.join(args.model_path_noseed, coarse_name, "point_cloud", f"iteration_{coarse_iter}", "point_cloud_0.ply")
        if os.path.exists(larger_motion_path):
            self.scene = Scene(dataset, self.gaussians, self.gaussians1, load_iteration=coarse_iter, load_path=larger_motion_path, is_train=True, coarse_name=coarse_name)
            print('Init gaussians from larger_motion_state.')
        else:
            print(f'Warning: {larger_motion_path} not found, using default initialization.')
            self.scene = Scene(dataset, self.gaussians, self.gaussians1, load_iteration=coarse_iter, is_train=True, coarse_name=coarse_name)

        self.init_logits_from_voxelized_joints()
        
        self.gaussians_orig = self.gaussians.copy()
        self.gaussians_orig.active_sh_degree = 3

        self.init_deform()
        self.gaussians.training_setup(opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        self.iteration = 1

        self.viewpoint_stacks = self.scene.train_cameras
        self.viewpoint_stacks_orig = self.scene.orig_cameras
        self.ema_loss_for_log = 0.0
        self.best_iteration = 15000
        self.best_joint_error = 1e10
        self.joint_metrics = []
        if not hasattr(self, "joint_types"):
            self.joint_types = ["s", "p", "r"]
            print(f'Initialized default joint types: {self.joint_types}')
            print(f'Joint structure: 1 static + {len(self.joint_types)-1} dynamic joints')

        self.progress_bar = tqdm.tqdm(range(self.iteration-1, opt.iterations), desc="Training progress")
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

        self.cd_loss_weight = args.cd_loss_weight
        self.metric_depth_loss_weight = args.metric_depth_loss_weight
        self.mono_depth_loss_weight = args.mono_depth_loss_weight
        
        self.e_step_interval = 4000
        self.m_step_interval = 4000
        self.current_phase = "E"
        self.e_step_count = 0
        self.m_step_count = 0
        self.last_phase_change = 0
        self.knn_mask_loss_weight = getattr(args, "knn_mask_loss_weight", 0.1)
        self.knn_mask_loss_k = getattr(args, "knn_mask_loss_k", 4)
        self.knn_mask_loss_sample = getattr(args, "knn_mask_loss_sample", 0)
        self._kdtree_cache = None
        self._kdtree_xyz_hash = None
        self._kdtree_sample_indices = None

        self.setup_em_optimization()

        print(
            f"EM optimization setup: E-step every {self.e_step_interval} iterations, "
            f"M-step every {self.m_step_interval} iterations"
        )
        print(f"Starting with {self.current_phase}-step")

    def setup_em_optimization(self):
        """Initialize EM schedule: start with E-step (mask)."""
        self.current_phase = "E"
        self.freeze_joint_parameters()
        self.unfreeze_mask_parameters()

    def freeze_joint_parameters(self):
        self.deform.set_joint_parameters_trainable(False)

    def unfreeze_joint_parameters(self):
        self.deform.set_joint_parameters_trainable(True)

    def freeze_mask_parameters(self):
        if hasattr(self.gaussians, "mask_logits"):
            self.gaussians.mask_logits.requires_grad_(False)

    def unfreeze_mask_parameters(self):
        if hasattr(self.gaussians, "mask_logits"):
            self.gaussians.mask_logits.requires_grad_(True)

    def check_phase_transition(self):
        if self.current_phase == "E":
            if self.iteration - self.last_phase_change >= self.e_step_interval:
                self.switch_to_m_step()
        else:
            if self.iteration - self.last_phase_change >= self.m_step_interval:
                self.switch_to_e_step()

    def switch_to_m_step(self):
        self.current_phase = "M"
        self.last_phase_change = self.iteration
        self.m_step_count += 1
        self.freeze_mask_parameters()
        self.unfreeze_joint_parameters()
        self.clear_knn_cache()

    def switch_to_e_step(self):
        self.current_phase = "E"
        self.last_phase_change = self.iteration
        self.e_step_count += 1
        self.freeze_joint_parameters()
        self.unfreeze_mask_parameters()

    def get_current_phase_info(self):
        if self.current_phase == "E":
            remaining = self.e_step_interval - (self.iteration - self.last_phase_change)
            return f"E-step (mask optimization), {remaining} iterations remaining"
        else:
            remaining = self.m_step_interval - (self.iteration - self.last_phase_change)
            return f"M-step (joint optimization), {remaining} iterations remaining"
    
    def visualize_sam_mask(self, mask, image_name, alpha_mask=None):
        torch.manual_seed(0)
        colors = torch.rand(50, 3, device=mask.device)
        mask_color = colors[mask % 50]
        mask_color = mask_color.permute(2, 0, 1)
        if alpha_mask is not None:
            mask_color = torch.cat([mask_color, alpha_mask], dim=0)
        torchvision.utils.save_image(mask_color, image_name)

    def init_logits_from_voxelized_joints(self):
        """Initialize mask logits and joint poses from voxelize_movable joint_voxel_info.npy."""
        summary_path = os.path.join(
            self.args.model_path_noseed,
            self.dataset.coarse_name,
            "point_cloud",
            f"iteration_{self.dataset.coarse_iter}",
            "joint_voxel_info.npy",
        )
        if not os.path.exists(summary_path):
            print(f"Joints summary not found: {summary_path}")
            return False
        
        summary_info = np.load(summary_path, allow_pickle=True).item()
        voxel_size = summary_info['voxel_size']
        shared_origin = np.array(summary_info['shared_origin'])
        
        gaussian_points = self.gaussians.get_xyz.detach().cpu().numpy()
        N = len(gaussian_points)
        K = self.gaussians.num_joints

        print(f"Found {len(summary_info['source_joints'])} voxelized joints with voxel_size={voxel_size}")
        print(f"Requested {K} joints (num_slots={self.gaussians.num_joints})")
        print(f"Expected: 1 static joint + {K-1} dynamic joints")

        logits = torch.zeros(N, K, device="cuda")
        logits[:, 0] = 2.0

        joints = torch.zeros(K, 7, device="cuda")
        joints[:, 0] = 1

        self.joint_types = ["s"]

        for dynamic_joint_idx in range(K - 1):
            voxel_info = summary_info["source_joints"][dynamic_joint_idx]
            source_voxels = set(map(tuple, voxel_info["source_voxels"]))

            point_voxel_indices = np.floor((gaussian_points - shared_origin) / voxel_size).astype(int)
            joint_inner_point_indices = [
                i
                for i, v_tuple in enumerate(map(tuple, point_voxel_indices))
                if v_tuple in source_voxels
            ]

            if len(joint_inner_point_indices) > 0:
                joint_inner_point_indices_t = torch.tensor(joint_inner_point_indices, device="cuda")
                logits[joint_inner_point_indices_t, 0] = -3.0
                logits[joint_inner_point_indices_t, dynamic_joint_idx + 1] = 3.0

                for j in range(1, K):
                    if j != dynamic_joint_idx + 1:
                        logits[joint_inner_point_indices_t, j] = -3.0

                print(f"Dynamic joint {dynamic_joint_idx+1}: Assigned {len(joint_inner_point_indices)} points.")
            joint_pose_info = summary_info["source_joints"][dynamic_joint_idx]["transformation_matrix"]
            R = joint_pose_info[:3, :3]
            t = joint_pose_info[:3, 3]
            R_tensor = torch.tensor(R, device="cuda", dtype=torch.float32)
            qr = matrix_to_quaternion(R_tensor)
            joints[dynamic_joint_idx + 1, :4] = qr
            joints[dynamic_joint_idx + 1, 4:7] = torch.tensor(t, device="cuda", dtype=torch.float32)
            print(
                f"Dynamic joint {dynamic_joint_idx+1} parameters initialized from transformation matrix"
            )

        self.gaussians.mask_logits = nn.Parameter(logits, requires_grad=True)

        if hasattr(self.deform.deform, "joints"):
            self.deform.deform.joints = nn.Parameter(joints, requires_grad=True)

        if hasattr(self.deform.deform, "joint_types"):
            if len(self.joint_types) < K:
                default_type = "r"
                self.joint_types.extend([default_type] * (K - len(self.joint_types)))
                print(f"Extended joint_types to {len(self.joint_types)} with default type '{default_type}'")

            self.deform.deform.joint_types = self.joint_types
            print(f"Joint types set: {self.joint_types}")
        else:
            print("Warning: deform model does not have 'joint_types' attribute")

        print("Successfully initialized logits and joint parameters from voxelized joints")
        print(f"Final joint types: {self.joint_types}")
        print(
            f"Total joints initialized: {len(self.joint_types)} "
            f"(1 static + {len(self.joint_types)-1} dynamic)"
        )
        return True
    
    def init_deform(self):
        self.deform.train_setting(self.opt)

    def compute_knn_mask_similarity_loss(self, k=4, sample_size=0):
        """KNN regularizer on mask logits in 3D (PyTorch3D CUDA knn_points)."""
        xyz = self.gaussians.get_xyz
        logits = self.gaussians.mask_logits
        N = xyz.shape[0]

        if isinstance(sample_size, int) and sample_size > 0 and sample_size < N:
            anchor_idx = torch.randperm(N, device=xyz.device)[:sample_size]
            anchors_xyz = xyz[anchor_idx]
            anchors_logits = logits[anchor_idx]
        else:
            anchors_xyz = xyz
            anchors_logits = logits

        k_conn = k + 1
        _, idx, _ = knn_points(
            anchors_xyz.unsqueeze(0),
            xyz.unsqueeze(0),
            K=k_conn,
            return_nn=False,
        )
        idx = idx.squeeze(0)
        neighbor_idx = idx[:, 1:]
        neighbor_logits = logits[neighbor_idx]
        anchors_logits_expanded = anchors_logits.unsqueeze(1)
        diffs = anchors_logits_expanded - neighbor_logits
        l2_dist = torch.sqrt(torch.clamp(diffs.pow(2).sum(dim=-1), min=1e-12))
        return l2_dist.mean()

    def clear_knn_cache(self):
        self._kdtree_cache = None
        self._kdtree_xyz_hash = None
        self._kdtree_sample_indices = None
    
    def compute_dynamic_joint_separation_loss(self, xt, k=8, margin=0.02, sample_size=0, use_soft_weight=True):
        """Hinge penalty on deformed positions so different dynamic joint labels stay separated (KNN on CPU)."""
        if not hasattr(self.gaussians, "joint_probs"):
            return torch.tensor(0.0, device=xt.device)
        joint_probs = self.gaussians.joint_probs  # [N, K]
        if joint_probs is None or joint_probs.numel() == 0:
            return torch.tensor(0.0, device=xt.device)
        pred_labels = joint_probs.argmax(dim=-1)
        pred_conf = joint_probs.max(dim=-1).values

        dynamic_mask = pred_labels > 0
        if dynamic_mask.sum() < 2:
            return torch.tensor(0.0, device=xt.device)
        dyn_idx = torch.nonzero(dynamic_mask, as_tuple=False).squeeze(1)
        dyn_pos = xt[dyn_idx]
        dyn_lab = pred_labels[dyn_idx]
        dyn_conf = pred_conf[dyn_idx]

        M = dyn_pos.shape[0]
        if isinstance(sample_size, int) and sample_size > 0 and sample_size < M:
            sel_idx = torch.randperm(M, device=dyn_pos.device)[:sample_size]
        else:
            sel_idx = torch.arange(M, device=dyn_pos.device)

        dyn_sel = dyn_pos[sel_idx]
        finite_mask = torch.isfinite(dyn_sel).all(dim=1)
        if finite_mask.sum() < 2:
            return torch.tensor(0.0, device=xt.device)
        sel_idx = sel_idx[finite_mask]
        dyn_sel = dyn_pos[sel_idx]

        if sel_idx.numel() <= 1:
            return torch.tensor(0.0, device=xt.device)

        pts_np = dyn_sel.detach().cpu().float().numpy()
        try:
            if pts_np.shape[0] < 2:
                return torch.tensor(0.0, device=xt.device)
            tree = KDTree(pts_np)
        except Exception:
            return torch.tensor(0.0, device=xt.device)

        kk = max(2, int(k))
        kk = min(kk, pts_np.shape[0])
        if kk < 2:
            return torch.tensor(0.0, device=xt.device)
        dists, inds = tree.query(pts_np, k=kk)

        inds_t = torch.from_numpy(inds).to(sel_idx.device).long()
        if inds_t.shape[1] > 1:
            inds_t = inds_t[:, 1:]
            dists = dists[:, 1:]
        if inds_t.numel() == 0:
            return torch.tensor(0.0, device=xt.device)

        labels_sel = dyn_lab[sel_idx]
        neighbor_labels = labels_sel[inds_t]
        diff_mask = neighbor_labels != labels_sel.unsqueeze(1)
        if not diff_mask.any():
            return torch.tensor(0.0, device=xt.device)

        dists_t = torch.from_numpy(dists.astype(np.float32)).to(xt.device)
        hinge = torch.clamp(margin - dists_t, min=0.0)

        if use_soft_weight:
            conf_sel = dyn_conf[sel_idx]
            neighbor_conf = conf_sel[inds_t]
            weight = conf_sel.unsqueeze(1) * neighbor_conf
        else:
            weight = torch.ones_like(dists_t)

        penal = hinge * weight * diff_mask.float()
        denom = diff_mask.float().sum().clamp_min(1.0)
        return penal.sum() / denom
    
    def train(self, iters=5000):
        for i in range(iters):
            self.train_step()

    def train_step(self):
        if self.iteration in [5000, 20000]:
            print("\n[ITER {}] Saving Checkpoint".format(self.iteration))
            torch.save(self.gaussians.capture(), self.args.model_path + "/chkpnt_{}.pth".format(self.iteration))

        self.check_phase_transition()

        if self.iteration % self.opt.oneupSHdegree_step == 0:
            self.gaussians.oneupSHdegree()
            
        id = randint(0, len(self.viewpoint_stacks) - 1)
        viewpoint_cam = self.viewpoint_stacks[id]
        id = randint(0, len(self.viewpoint_stacks_orig) - 1)
        viewpoint_cam_orig = self.viewpoint_stacks_orig[id]
        random_bg = (
            not self.dataset.white_background
            and self.opt.random_bg_color
            and viewpoint_cam.gt_alpha_mask is not None
        )
        bg = self.background if not random_bg else torch.rand_like(self.background).cuda()
        if hasattr(self.deform.deform, "joint_types"):
            joint_types = self.deform.deform.joint_types
        else:
            joint_types = self.joint_types

        d_values = self.deform.deform.one_transform(
            self.gaussians,
            joint_types[1:],
            is_training=True,
            current_phase=self.current_phase,
        )
        d_xyz, d_rot = d_values["d_xyz"], d_values["d_rotation"]

        render_pkg_re = render(viewpoint_cam, self.gaussians, self.pipe, bg, d_xyz=d_xyz, d_rot=d_rot)
        render_pkg_re1 = render(viewpoint_cam, self.gaussians1, self.pipe, bg)
        render_pkg_re_orig = render(viewpoint_cam_orig, self.gaussians, self.pipe, bg)
        image = render_pkg_re["render"]
        viewspace_point_tensor = render_pkg_re["viewspace_points"]
        visibility_filter = render_pkg_re["visibility_filter"]
        radii = render_pkg_re["radii"]
        gt_image = viewpoint_cam.original_image.cuda()
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        if random_bg:
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * bg[:, None, None]
        elif self.dataset.white_background and viewpoint_cam.gt_alpha_mask is not None:
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * self.background[:, None, None]

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if self.current_phase == "E":
            tau_conf = getattr(self.args, "e_gate_pmax", 0.5)
            tau_gap = getattr(self.args, "e_gate_gap", 0.2)
            min_area = getattr(self.args, "e_gate_area", 50)
            temp = getattr(self.args, "e_soft_temp", 1.0)

            gt_sam_mask = viewpoint_cam_orig.mask.to(torch.int64)
            gt_alpha_mask = viewpoint_cam_orig.gt_alpha_mask.to(torch.long)
            ren_mask = render_pkg_re_orig["mask_logits"]
            ren_prob = torch.softmax(ren_mask / max(1e-6, temp), dim=0)
            new_joint_mask = torch.argmax(ren_mask, dim=0).to(torch.long).detach().clone()
            valid_pixels_mask = (gt_alpha_mask == 1).squeeze(0)

            num_valid = valid_pixels_mask.sum()
            if num_valid > 0:
                C = ren_prob.shape[0]
                sam_ids = gt_sam_mask[valid_pixels_mask]
                probs = ren_prob[:, valid_pixels_mask]
                p_max, _ = probs.max(dim=0)
                top2_vals, _ = probs.topk(k=min(2, C), dim=0)
                gap = top2_vals[0] - (top2_vals[1] if C > 1 else 0.0)

                unique_sam_ids, sam_idx = torch.unique(sam_ids, sorted=False, return_inverse=True)
                U = unique_sam_ids.numel()

                counts = torch.zeros(U, C, device=ren_prob.device, dtype=ren_prob.dtype)
                counts.index_add_(0, sam_idx, probs.T)

                area = torch.zeros(U, device=ren_prob.device, dtype=torch.long)
                area.index_add_(0, sam_idx, torch.ones_like(sam_idx, dtype=torch.long))
                pmax_sum = torch.zeros(U, device=ren_prob.device)
                pmax_sum.index_add_(0, sam_idx, p_max)
                gap_sum = torch.zeros(U, device=ren_prob.device)
                gap_sum.index_add_(0, sam_idx, gap)
                pmax_mean = pmax_sum / torch.clamp(area.to(pmax_sum.dtype), min=1)
                gap_mean = gap_sum / torch.clamp(area.to(gap_sum.dtype), min=1)

                C = ren_prob.shape[0]
                jt = self.deform.deform.joint_types
                boost_prismatic_class = torch.zeros(C, dtype=torch.bool, device=ren_prob.device)
                for ci in range(min(C, len(jt))):
                    if jt[ci] == "p":
                        boost_prismatic_class[ci] = True
                p_vote_boost = getattr(self.args, "e_prismatic_vote_boost", 4)
                vote_weights = torch.ones(C, dtype=counts.dtype, device=counts.device)
                vote_weights[boost_prismatic_class] = p_vote_boost
                weighted_counts = counts * vote_weights.unsqueeze(0)
                majority_class_per_sam = weighted_counts.argmax(dim=1)

                allow_update = (pmax_mean > tau_conf) & (gap_mean > tau_gap) & (area >= min_area)

                allow_pix = allow_update[sam_idx]
                allow_pix_full = torch.zeros_like(valid_pixels_mask, dtype=torch.bool)
                allow_pix_full[valid_pixels_mask] = allow_pix
                per_pixel_major = majority_class_per_sam[sam_idx]
                new_joint_mask[allow_pix_full] = per_pixel_major[allow_pix]

            num_classes = ren_mask.shape[0]
            target = new_joint_mask.clone()
            target[~valid_pixels_mask] = -1
            ce_weights = torch.ones(num_classes, device=ren_mask.device)
            ce_weights[0] = getattr(self.args, "mask_ce_w_static", 1.2)
            mask_criterion = nn.CrossEntropyLoss(weight=ce_weights, ignore_index=-1)
            mask_loss = mask_criterion(ren_mask.unsqueeze(0), target.unsqueeze(0))
            loss = loss + 0.1 * mask_loss

            if self.gaussians.mask_logits.argmax(dim=-1).bincount()[1:].sum() < 3000:
                knn_loss = 0
            else:
                knn_loss = self.compute_knn_mask_similarity_loss(
                    k=self.knn_mask_loss_k,
                    sample_size=self.knn_mask_loss_sample
                )
            loss = loss + self.knn_mask_loss_weight * knn_loss

        depth_loss = torch.tensor([0.0])
        if self.metric_depth_loss_weight > 0:
            depth = render_pkg_re["depth"]
            gt_depth = viewpoint_cam.depth.cuda() if viewpoint_cam.depth is not None else render_pkg_re1["depth"]
            invalid_mask = (gt_depth < 0.1) & (gt_alpha_mask > 0.5)
            valid_mask = ~invalid_mask
            n_valid_pixel = valid_mask.sum()
            if n_valid_pixel > 100:
                depth_loss = (torch.log(1 + torch.abs(depth - gt_depth)) * valid_mask).sum() / n_valid_pixel
                loss = loss + depth_loss * self.metric_depth_loss_weight
        render_pkg_ori = render(viewpoint_cam, self.gaussians_orig, self.pipe, bg)
        render_pkg_re = render(viewpoint_cam, self.gaussians, self.pipe, bg)
        orig_rgb = render_pkg_ori["render"]
        gs_rgb = render_pkg_re["render"]
        orig_depth = render_pkg_ori["depth"]
        gs_depth = render_pkg_re["depth"]
        Ll1_rgb = l1_loss(orig_rgb, gs_rgb)
        loss = loss + (1.0 - self.opt.lambda_dssim) * Ll1_rgb + self.opt.lambda_dssim * (1.0 - ssim(orig_rgb, gs_rgb))
        
        valid_mask = (gs_depth > 0.1) & (gt_alpha_mask > 0.5)
        n_valid_pixel = valid_mask.sum()
        if n_valid_pixel > 100:
            depth_loss = (torch.log(1 + torch.abs(orig_depth - gs_depth)) * valid_mask).sum() / n_valid_pixel
            depth_loss = depth_loss.mean() * self.metric_depth_loss_weight
        loss = loss + depth_loss
        
        loss.backward()
    
        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                postfix_dict = {
                    "Loss": f"{self.ema_loss_for_log:.{6}f}",
                    "Phase": self.current_phase,
                    "E_count": self.e_step_count,
                    "M_count": self.m_step_count,
                }
                self.progress_bar.set_postfix(postfix_dict)
                self.progress_bar.update(10)
                
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()

            if self.iteration % 10 == 0 or self.iteration == 1:
                try:
                    pred_joint_list = self.deform.deform.get_joint_param()
                    gt_info_list = read_gt(
                        os.path.expanduser(f"{self.args.source_path}/gt/trans.json")
                    )
                    reverse = self.larger_motion == "1"
                    joint_type_psuedo = ["p", "p", "r", "r", "r", "r"]
                    self.joint_metrics, real_perm = eval_axis_and_state_all(
                        pred_joint_list, joint_type_psuedo, gt_info_list, reverse=reverse
                    )
                    with open(os.path.join(self.args.model_path, "joint_metrics.json"), "w") as f:
                        json.dump(self.joint_metrics, f)
                    with open(os.path.join(self.args.model_path, "pred_joint_list.json"), "w") as f:
                        json.dump(pred_joint_list, f)
                except Exception:
                    pass
            if self.iteration % 100 == 0 and self.iteration > 1500:
                cur_joint_error = sum([sum(m) for m in self.joint_metrics]) if len(self.joint_metrics) > 0 else 1e5
                if cur_joint_error < self.best_joint_error or (self.iteration == self.args.iterations and self.best_iteration <= 15000):
                    self.best_iteration = self.iteration
                    self.best_joint_error = cur_joint_error

            if self.iteration in self.saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration)
                self.deform.save_weights(self.args.model_path, self.iteration)
            if self.iteration == self.best_iteration:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration, is_best=True)
                self.deform.save_weights(self.args.model_path, self.iteration, is_best=True)
            
            # Keep track of max radii in image-space for pruning
            if self.gaussians.max_radii2D.shape[0] == 0:
                self.gaussians.max_radii2D = torch.zeros_like(radii)
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            if self.iteration in [10000]:
                threshould = 10.0
                current_joint_types = self.deform.deform.joint_types
                joint_params = self.deform.deform.get_joint_param()
                target_joint_types = ["s"]
                for dynamic_joint_idx in range(len(joint_params)):
                    theta = joint_params[dynamic_joint_idx]["theta"]
                    if current_joint_types[dynamic_joint_idx + 1] == "r":
                        target_joint_types.append("p" if theta < threshould else "r")
                    else:
                        target_joint_types.append("p")

                print(f"Current joint types: {current_joint_types}")
                print(f"Target joint types: {target_joint_types}")

                success = self.deform.switch_joint_types(target_joint_types, self.opt)
                if success:
                    print("Successfully switched joint types")
                else:
                    print("Failed to switch joint types")
            if self.iteration < self.opt.densify_until_iter and self.current_phase == "E":
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.opacity_cull, self.scene.cameras_extent, size_threshold)
                
                if self.iteration % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            self.gaussians.optimizer.step()
            self.gaussians.update_learning_rate(self.iteration)
            self.gaussians.optimizer.zero_grad(set_to_none=True)

            self.deform.optimizer.step()
            self.deform.optimizer.zero_grad()
            self.deform.update_learning_rate(self.iteration)

        self.iteration += 1

    def visualize(self, image, gt_image, gt_depth, depth):
        torchvision.utils.save_image(image.detach(), "img.png")
        torchvision.utils.save_image(gt_image, "img_gt.png")
        torchvision.utils.save_image(vis_depth(gt_depth), "gt.png")
        torchvision.utils.save_image(vis_depth(depth.detach()), "pred.png")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5000, 10_000, 15_000, 20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--coarse_name', type=str, default='coarse_point_cloud', help='Name of the coarse point cloud')
    parser.add_argument('--coarse_iter', type=int, default=0, help='Iteration of the coarse point cloud')
    args = parser.parse_args(sys.argv[1:])
    args.source_path = f"{args.source_path}/{args.dataset}/{args.subset}/{args.scene_name}"
    args.model_path_noseed = os.path.join("outputs", args.dataset, args.subset, args.scene_name)
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    seed_everything(args.seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    jt_path = os.path.join(os.path.dirname(__file__), "arguments", "joint_types_pcd.json")
    with open(jt_path, "r", encoding="utf-8") as f:
        args.joint_types = json.load(f)[args.dataset][args.subset][args.scene_name]
    args.num_slots = len(args.joint_types.split(","))
    print(
        f"Joint types from {jt_path}: num_slots={args.num_slots}, "
        f"types={args.joint_types} (1 static + {args.num_slots - 1} dynamic)"
    )
    trainer = Trainer(
        args=args,
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        saving_iterations=args.save_iterations,
        coarse_name=args.coarse_name,
        coarse_iter=args.coarse_iter,
    )
    trainer.train(args.iterations)
    print("\nTraining complete.")