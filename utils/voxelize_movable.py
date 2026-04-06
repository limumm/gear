import argparse
import json
import os
import sys
from collections import deque

import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Voxel-based movable-part segmentation (Top-K connected components).

- Runs connected-component analysis and keeps the k largest regions.
- Works best when articulated parts are spatially separated.
- Preserves geometric connectivity of each part.

Using saved voxel data for mask assignment:

```python
voxel_info = load_voxel_info("path/to/joint_voxel_info.npy")
joint_masks = create_joint_masks_from_voxel_info(voxel_info, point_cloud_points)
source_indices, target_indices = get_joint_point_indices_from_voxel_info(
    voxel_info, joint_id=0
)
for i, mask in enumerate(joint_masks):
    joint_points = point_cloud_points[mask]
    print(f"Joint {i}: {len(joint_points)} points")
```
"""

# Minimum PCA box aspect ratio (max edge / min edge) for coarse pose gating
ASPECT_RATIO_THRESHOLD = 3.0

# Set by extract_dynamic_joints(verbose=...); when True, print diagnostics and show Open3D windows
_VOXEL_VERBOSE = False


def _voxel_vprint(*args, **kwargs):
    if _VOXEL_VERBOSE:
        print(*args, **kwargs)


def component_aspect_ratio(component_voxels, voxel_size, shared_origin):
    """
    PCA-aligned box extents in 3D; returns (max/min ratio, max_edge, min_edge).
    """
    if component_voxels is None or len(component_voxels) == 0:
        return 0.0, 0.0, 0.0
    pts = []
    for v_idx in component_voxels:
        center = (np.array(v_idx) + 0.5) * voxel_size + shared_origin
        pts.append(center)
    pts_np = np.array(pts)
    if pts_np.shape[0] < 2:
        return 0.0, 0.0, 0.0
    centered = pts_np - np.mean(pts_np, axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    proj3 = centered @ evecs  # [N,3]
    extents = []
    for ax in range(3):
        vals = proj3[:, ax]
        length = (vals.max() - vals.min()) if vals.size > 0 else 0.0
        extents.append(length)
    max_edge = float(np.max(extents)) if extents else 0.0
    min_edge = float(np.min(extents)) if extents else 0.0
    min_edge = max(1e-8, min_edge)
    ratio = max_edge / min_edge
    return ratio, max_edge, min_edge

def get_angle_axis_from_matrix(transformation_matrix):
    """
    Extract rotation angle (rad) and axis from a 4x4 rigid transform.

    Args:
        transformation_matrix: 4x4 homogeneous transform.

    Returns:
        angle: rotation angle in radians.
        axis: unit rotation axis [x, y, z].
    """
    rotation_matrix = transformation_matrix[:3, :3]

    trace = np.trace(rotation_matrix)
    cos_theta = (trace - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    if abs(cos_theta - 1) < 1e-6:
        return 0.0, np.array([0, 0, 1])
    elif abs(cos_theta + 1) < 1e-6:
        angle = np.pi
        K = (rotation_matrix + np.identity(3)) / 2
        for i in range(3):
            if np.any(K[i, :] != 0):
                axis = K[i, :]
                axis = axis / np.linalg.norm(axis)
                return angle, axis
        return angle, np.array([0, 0, 1])
    else:
        angle = np.arccos(cos_theta)
        K = (rotation_matrix - rotation_matrix.T) / (2 * np.sin(angle))
        axis = np.array([K[2, 1], K[0, 2], K[1, 0]])
        axis = axis / np.linalg.norm(axis)
        return angle, axis
    
def create_rotation_matrix_from_axis_angle(axis, angle):
    """
    Build 3x3 rotation from axis-angle (Rodrigues): R = I + sin(θ)K + (1-cos(θ))K².

    Args:
        axis: rotation axis [x, y, z].
        angle: angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    K = np.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    
    I = np.identity(3)
    rotation_matrix = (I + 
                      np.sin(angle) * K + 
                      (1 - np.cos(angle)) * (K @ K))
    
    return rotation_matrix

def calculate_voxel_overlap_after_transform(source_voxels, target_voxels, transformation_matrix, voxel_size, shared_origin):
    """
    Count voxels of source that map into target voxels after applying the transform.

    Returns:
        overlap_count: number of overlapping voxels.
        overlap_ratio: overlap_count / min(|source|, |target|).
    """
    if not source_voxels or not target_voxels:
        return 0, 0.0

    source_points = []
    for v_idx in source_voxels:
        center = (np.array(v_idx) + 0.5) * voxel_size + shared_origin
        source_points.append(center)
    source_points_np = np.array(source_points)

    if not np.allclose(transformation_matrix, np.eye(4)):
        source_points_homo = np.hstack([source_points_np, np.ones((len(source_points_np), 1))])
        transformed_points_homo = (transformation_matrix @ source_points_homo.T).T
        transformed_points = transformed_points_homo[:, :3]
    else:
        transformed_points = source_points_np

    transformed_voxel_indices = np.floor((transformed_points - shared_origin) / voxel_size).astype(int)
    transformed_voxels = set(map(tuple, transformed_voxel_indices))

    overlap_voxels = transformed_voxels & target_voxels
    overlap_count = len(overlap_voxels)

    min_size = min(len(source_voxels), len(target_voxels))
    overlap_ratio = overlap_count / min_size if min_size > 0 else 0.0
    
    return overlap_count, overlap_ratio

def match_voxel_components(source_components, target_components, voxel_size, shared_origin):
    """
    Match source/target voxel components via plane-based transform and overlap scoring.

    Returns:
        List of (source_idx, target_idx) pairs.
    """
    matched_pairs = []
    used_target_indices = set()

    _voxel_vprint("Matching voxel components (plane transform + overlap)...")

    aspect_threshold = ASPECT_RATIO_THRESHOLD
    source_aspect = [ component_aspect_ratio(comp, voxel_size, shared_origin) for comp in source_components ]
    target_aspect = [ component_aspect_ratio(comp, voxel_size, shared_origin) for comp in target_components ]

    for i, source_comp in enumerate(source_components):
        _voxel_vprint(f"\nSource component {i+1}/{len(source_components)}...")

        best_match_idx = -1
        best_overlap_count = 0
        best_overlap_ratio = 0.0
        best_transformation = np.eye(4)
        s_ratio, s_long, s_short = source_aspect[i]
        if s_ratio < aspect_threshold:
            _voxel_vprint(
                f"  Skip coarse pose: source aspect ratio {s_ratio:.2f} < {aspect_threshold:.2f}"
            )
            continue

        for j, target_comp in enumerate(target_components):
            if j in used_target_indices:
                continue

            _voxel_vprint(f"  Try target component {j+1}...")
            t_ratio, t_long, t_short = target_aspect[j]
            if t_ratio < aspect_threshold:
                _voxel_vprint(
                    f"    Skip target: aspect ratio {t_ratio:.2f} < {aspect_threshold:.2f}"
                )
                continue

            source_plane, target_plane = fit_planes_from_matched_voxels(
                source_comp, target_comp, voxel_size, shared_origin
            )
            
            if source_plane is None or target_plane is None:
                _voxel_vprint("    Plane fit failed, skip")
                continue

            transformation_matrix = calculate_transform_from_planes(
                source_plane, target_plane, source_comp, target_comp, voxel_size, shared_origin
            )

            overlap_count, overlap_ratio = calculate_voxel_overlap_after_transform(
                source_comp, target_comp, transformation_matrix, voxel_size, shared_origin
            )

            _voxel_vprint(f"    Overlap voxels: {overlap_count}, ratio: {overlap_ratio:.3f}")

            if overlap_count > best_overlap_count:
                best_overlap_count = overlap_count
                best_overlap_ratio = overlap_ratio
                best_match_idx = j
                best_transformation = transformation_matrix
        
        if best_match_idx != -1:
            matched_pairs.append((i, best_match_idx))
            used_target_indices.add(best_match_idx)
            _voxel_vprint(
                f"Source {i+1} -> target {best_match_idx+1} "
                f"(overlap {best_overlap_count}, ratio {best_overlap_ratio:.3f})"
            )
        else:
            _voxel_vprint(f"Source {i+1}: no matching target component")
    
    return matched_pairs

def fit_planes_from_matched_voxels(source_voxels, target_voxels, voxel_size, shared_origin):
    """
    Fit a plane to source voxels and one to target voxels (PCA smallest eigenvector as normal).

    Returns:
        ((source_normal, source_point), (target_normal, target_point)) or (None, None).
    """
    if len(source_voxels) < 3 or len(target_voxels) < 3:
        _voxel_vprint("Too few voxels to fit planes")
        return None, None

    source_points = []
    for v_idx in source_voxels:
        center = (np.array(v_idx) + 0.5) * voxel_size + shared_origin
        source_points.append(center)
    source_points_np = np.array(source_points)
    
    target_points = []
    for v_idx in target_voxels:
        center = (np.array(v_idx) + 0.5) * voxel_size + shared_origin
        target_points.append(center)
    target_points_np = np.array(target_points)

    source_centroid = np.mean(source_points_np, axis=0)
    source_centered = source_points_np - source_centroid
    source_cov = np.cov(source_centered.T)
    source_eigenvals, source_eigenvecs = np.linalg.eigh(source_cov)
    source_normal = source_eigenvecs[:, np.argmin(source_eigenvals)]
    source_normal /= np.linalg.norm(source_normal)

    target_centroid = np.mean(target_points_np, axis=0)
    target_centered = target_points_np - target_centroid
    target_cov = np.cov(target_centered.T)
    target_eigenvals, target_eigenvecs = np.linalg.eigh(target_cov)
    target_normal = target_eigenvecs[:, np.argmin(target_eigenvals)]
    target_normal /= np.linalg.norm(target_normal)
    
    return (source_normal, source_centroid), (target_normal, target_centroid)

def calculate_transform_from_planes(source_plane, target_plane, source_voxels=None, target_voxels=None, voxel_size=None, shared_origin=None):
    """
    Rigid transform aligning two planes: rotation about their line of intersection.
    If voxels are given, pick 0° vs 180° by overlap.
    """
    if source_plane is None or target_plane is None:
        _voxel_vprint("Invalid planes, return identity")
        return np.eye(4)
    
    source_normal, source_point = source_plane
    target_normal, target_point = target_plane
    
    dot_product = np.abs(np.dot(source_normal, target_normal))
    if dot_product > 0.95:
        _voxel_vprint("Planes nearly parallel, return identity")
        return np.eye(4)

    rotation_axis = np.cross(source_normal, target_normal)
    axis_norm = np.linalg.norm(rotation_axis)
    
    if axis_norm < 1e-6:
        _voxel_vprint("Cannot determine rotation axis, return identity")
        return np.eye(4)

    rotation_axis /= axis_norm

    A = np.vstack([source_normal, target_normal])
    b = np.array([np.dot(source_point, source_normal), np.dot(target_point, target_normal)])

    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, b)

    try:
        rotation_center = np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        _voxel_vprint("Normal equations failed, use pseudoinverse")
        rotation_center = np.dot(np.linalg.pinv(ATA), ATb)

    cos_angle = np.clip(np.dot(source_normal, target_normal), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if source_voxels is not None and target_voxels is not None and voxel_size is not None and shared_origin is not None:
        _voxel_vprint("Compare overlap for 0° vs 180° rotation...")

        rotation_matrix_0 = create_rotation_matrix_from_axis_angle(rotation_axis, angle)
        T1 = np.eye(4)
        T1[:3, 3] = -rotation_center
        R0 = np.eye(4)
        R0[:3, :3] = rotation_matrix_0
        T2 = np.eye(4)
        T2[:3, 3] = rotation_center
        transform_0 = T2 @ R0 @ T1

        rotation_matrix_180 = create_rotation_matrix_from_axis_angle(rotation_axis, angle + np.pi)
        R180 = np.eye(4)
        R180[:3, :3] = rotation_matrix_180
        transform_180 = T2 @ R180 @ T1

        overlap_0, ratio_0 = calculate_voxel_overlap_after_transform(
            source_voxels, target_voxels, transform_0, voxel_size, shared_origin
        )
        overlap_180, ratio_180 = calculate_voxel_overlap_after_transform(
            source_voxels, target_voxels, transform_180, voxel_size, shared_origin
        )

        _voxel_vprint(f"  0° overlap: {overlap_0} voxels, ratio {ratio_0:.3f}")
        _voxel_vprint(f"  180° overlap: {overlap_180} voxels, ratio {ratio_180:.3f}")

        if overlap_180 > overlap_0:
            _voxel_vprint("  Use 180° (higher overlap)")
            transformation_matrix = transform_180
            final_angle = angle + np.pi
        else:
            _voxel_vprint("  Use 0° (higher overlap)")
            transformation_matrix = transform_0
            final_angle = angle
    else:
        rotation_matrix = create_rotation_matrix_from_axis_angle(rotation_axis, angle)
        
        T1 = np.eye(4)
        T1[:3, 3] = -rotation_center
        R = np.eye(4)
        R[:3, :3] = rotation_matrix
        T2 = np.eye(4)
        T2[:3, 3] = rotation_center
        
        transformation_matrix = T2 @ R @ T1
        final_angle = angle

    _voxel_vprint("Computed transform:")
    _voxel_vprint(f"  Intersection axis: {np.round(rotation_axis, 3)}")
    _voxel_vprint(f"  Rotation center: {np.round(rotation_center, 3)}")
    _voxel_vprint(f"  Rotation angle: {np.degrees(final_angle):.2f} deg")
    
    return transformation_matrix

def validate_transformation_matrix(transformation_matrix, source_voxels, target_voxels, voxel_size, shared_origin):
    """Check rotation part is proper orthogonal (det≈1, R.T@R≈I)."""
    if np.allclose(transformation_matrix, np.eye(4)):
        return True, "identity (no transform)"

    det = np.linalg.det(transformation_matrix[:3, :3])
    if abs(det - 1.0) > 0.1:
        return False, f"bad rotation det: {det:.3f}"

    R = transformation_matrix[:3, :3]
    should_be_identity = R.T @ R
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-3):
        return False, "not a valid rotation matrix"

    return True, "valid transform"

def _print_joint_brief_summary(joint_voxel_info):
    """One-line summary (always printed)."""
    joints = joint_voxel_info["source_joints"]
    n = len(joints)
    matched = sum(1 for j in joints if j.get("matched_target_idx", -1) >= 0)
    valid = sum(
        1
        for j in joints
        if j.get("transform_valid", False) and j.get("overlap_ratio", 0) >= 0.1
    )
    print(f"[voxelize] slots={n}, matched={matched}, valid_transform(overlap>=0.1)={valid}")


def print_transformation_summary(joint_voxel_info):
    """Verbose per-joint transform summary."""
    _voxel_vprint("\n" + "=" * 70)
    _voxel_vprint("Joint transformation summary")
    _voxel_vprint("=" * 70)

    for i, joint_info in enumerate(joint_voxel_info["source_joints"]):
        _voxel_vprint(f"\nJoint {i+1}:")
        _voxel_vprint(f"  matched_target_idx: {joint_info['matched_target_idx']}")
        _voxel_vprint(f"  overlap_count: {joint_info.get('overlap_count', 0)}")
        _voxel_vprint(f"  overlap_ratio: {joint_info.get('overlap_ratio', 0.0):.3f}")
        _voxel_vprint(f"  transform_valid: {joint_info.get('transform_valid', False)}")
        _voxel_vprint(f"  info: {joint_info.get('transform_error_info', '')}")

        if not np.allclose(joint_info["transformation_matrix"], np.eye(4)):
            t = joint_info["transformation_matrix"][:3, 3]

            try:
                angle, axis = get_angle_axis_from_matrix(joint_info["transformation_matrix"])
                _voxel_vprint(f"  angle: {np.degrees(angle):.2f} deg")
                _voxel_vprint(f"  axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
                _voxel_vprint(f"  translation: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
            except Exception:
                _voxel_vprint("  could not parse rotation from matrix")
        else:
            _voxel_vprint("  transform: identity")

def visualize_transformed_matching(joint_voxel_info, source_components, target_components, voxel_size, shared_origin, save_dir=None):
    """Verbose Open3D view: source, transformed source, target, overlap (yellow)."""
    _voxel_vprint("\n" + "=" * 50)
    _voxel_vprint("Visualize transform matching...")
    _voxel_vprint("=" * 50)

    for i, joint_info in enumerate(joint_voxel_info["source_joints"]):
        if joint_info["matched_target_idx"] == -1:
            _voxel_vprint(f"Joint {i+1} unmatched, skip viz")
            continue

        _voxel_vprint(f"Joint {i+1} matching viz...")

        source_voxels = source_components[i]
        target_voxels = target_components[joint_info["matched_target_idx"]]
        transformation_matrix = joint_info["transformation_matrix"]

        source_points = []
        for v_idx in source_voxels:
            center = (np.array(v_idx) + 0.5) * voxel_size + shared_origin
            source_points.append(center)
        source_points_np = np.array(source_points)

        target_points = []
        for v_idx in target_voxels:
            center = (np.array(v_idx) + 0.5) * voxel_size + shared_origin
            target_points.append(center)
        target_points_np = np.array(target_points)

        if not np.allclose(transformation_matrix, np.eye(4)):
            source_points_homo = np.hstack([source_points_np, np.ones((len(source_points_np), 1))])
            transformed_points_homo = (transformation_matrix @ source_points_homo.T).T
            transformed_points = transformed_points_homo[:, :3]
        else:
            transformed_points = source_points_np

        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points_np)
        source_pcd.paint_uniform_color([1, 0, 0])

        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
        transformed_pcd.paint_uniform_color([0, 0, 1])

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points_np)
        target_pcd.paint_uniform_color([0, 1, 0])

        transformed_voxel_indices = np.floor((transformed_points - shared_origin) / voxel_size).astype(int)
        transformed_voxels = set(map(tuple, transformed_voxel_indices))
        overlap_voxels = transformed_voxels & target_voxels

        overlap_points = []
        for v_idx in overlap_voxels:
            center = (np.array(v_idx) + 0.5) * voxel_size + shared_origin
            overlap_points.append(center)
        
        overlap_pcd = o3d.geometry.PointCloud()
        if overlap_points:
            overlap_pcd.points = o3d.utility.Vector3dVector(np.array(overlap_points))
            overlap_pcd.paint_uniform_color([1, 1, 0])

        geometries = [source_pcd, transformed_pcd, target_pcd]
        if overlap_points:
            geometries.append(overlap_pcd)
        
        _voxel_vprint(f"  source (red): {len(source_points_np)} pts")
        _voxel_vprint(f"  transformed (blue): {len(transformed_points)} pts")
        _voxel_vprint(f"  target (green): {len(target_points_np)} pts")
        _voxel_vprint(f"  overlap (yellow): {len(overlap_points)} pts")
        _voxel_vprint(f"  overlap_ratio: {joint_info.get('overlap_ratio', 0.0):.3f}")

        if _VOXEL_VERBOSE:
            o3d.visualization.draw_geometries(
                geometries,
                window_name=f"Joint {i+1} match",
                width=1200,
                height=800,
            )

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            _voxel_vprint(f"  viz output dir: {save_dir}")

    _voxel_vprint("Transform matching visualization done.")

def save_combined_matched_voxels(joint_voxel_info, source_components, target_components, voxel_size, shared_origin, save_dir):
    """Merge valid matched voxel clouds and write combined_all_voxels.ply."""
    _voxel_vprint("\n" + "=" * 50)
    _voxel_vprint("Merge matched voxel point clouds...")
    _voxel_vprint("=" * 50)

    if not save_dir:
        _voxel_vprint("No save_dir, skip merge")
        return

    os.makedirs(save_dir, exist_ok=True)

    all_source_points = []
    all_transformed_points = []
    all_target_points = []
    all_overlap_points = []

    colors = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [1, 0, 0.5], [0.5, 1, 0], [0, 0.5, 1]
    ]
    
    matched_joints = 0

    for i, joint_info in enumerate(joint_voxel_info["source_joints"]):
        if joint_info["matched_target_idx"] == -1:
            _voxel_vprint(f"Joint {i+1} unmatched, skip")
            continue

        if not (
            joint_info.get("transform_valid", False) and joint_info.get("overlap_ratio", 0) >= 0.1
        ):
            _voxel_vprint(
                f"Joint {i+1} invalid match (overlap {joint_info.get('overlap_ratio', 0):.3f}, "
                f"valid {joint_info.get('transform_valid', False)}), skip"
            )
            continue

        _voxel_vprint(f"Joint {i+1} matched voxels...")

        source_voxels = source_components[i]
        target_voxels = target_components[joint_info["matched_target_idx"]]
        transformation_matrix = joint_info["transformation_matrix"]

        source_points = []
        for v_idx in source_voxels:
            center = (np.array(v_idx) + 0.5) * voxel_size + shared_origin
            source_points.append(center)
        source_points_np = np.array(source_points)

        target_points = []
        for v_idx in target_voxels:
            center = (np.array(v_idx) + 0.5) * voxel_size + shared_origin
            target_points.append(center)
        target_points_np = np.array(target_points)

        if not np.allclose(transformation_matrix, np.eye(4)):
            source_points_homo = np.hstack([source_points_np, np.ones((len(source_points_np), 1))])
            transformed_points_homo = (transformation_matrix @ source_points_homo.T).T
            transformed_points = transformed_points_homo[:, :3]
        else:
            transformed_points = source_points_np

        transformed_voxel_indices = np.floor((transformed_points - shared_origin) / voxel_size).astype(int)
        transformed_voxels = set(map(tuple, transformed_voxel_indices))
        overlap_voxels = transformed_voxels & target_voxels

        overlap_points = []
        for v_idx in overlap_voxels:
            center = (np.array(v_idx) + 0.5) * voxel_size + shared_origin
            overlap_points.append(center)

        joint_color = colors[i % len(colors)]

        all_source_points.extend(source_points_np)
        all_transformed_points.extend(transformed_points)
        all_target_points.extend(target_points_np)
        all_overlap_points.extend(overlap_points)
        
        matched_joints += 1
        _voxel_vprint(
            f"  joint {i+1}: src {len(source_points_np)} xf {len(transformed_points)} "
            f"tgt {len(target_points_np)} ov {len(overlap_points)}"
        )

    if matched_joints == 0:
        _voxel_vprint("No matched joints, cannot merge")
        return

    has_valid_matches = False
    for joint_info in joint_voxel_info["source_joints"]:
        if (
            joint_info.get("matched_target_idx", -1) != -1
            and joint_info.get("transform_valid", False)
            and joint_info.get("overlap_ratio", 0) >= 0.1
        ):
            has_valid_matches = True
            break

    if not has_valid_matches:
        _voxel_vprint("No valid matches, skip combined_all_voxels.ply")
        return

    _voxel_vprint(f"\nMerge {matched_joints} matched joints...")

    combined_source_pcd = o3d.geometry.PointCloud()
    combined_source_pcd.points = o3d.utility.Vector3dVector(np.array(all_source_points))
    combined_source_pcd.paint_uniform_color([1, 0, 0])

    combined_transformed_pcd = o3d.geometry.PointCloud()
    combined_transformed_pcd.points = o3d.utility.Vector3dVector(np.array(all_transformed_points))
    combined_transformed_pcd.paint_uniform_color([0, 0, 1])

    combined_target_pcd = o3d.geometry.PointCloud()
    combined_target_pcd.points = o3d.utility.Vector3dVector(np.array(all_target_points))
    combined_target_pcd.paint_uniform_color([0, 1, 0])

    combined_overlap_pcd = o3d.geometry.PointCloud()
    if all_overlap_points:
        combined_overlap_pcd.points = o3d.utility.Vector3dVector(np.array(all_overlap_points))
        combined_overlap_pcd.paint_uniform_color([1, 1, 0])

    all_points_with_colors = []
    all_colors = []
    
    for i, joint_info in enumerate(joint_voxel_info["source_joints"]):
        if joint_info["matched_target_idx"] == -1:
            continue

        if not (
            joint_info.get("transform_valid", False) and joint_info.get("overlap_ratio", 0) >= 0.1
        ):
            continue

        joint_color = colors[i % len(colors)]

        source_voxels = source_components[i]
        target_voxels = target_components[joint_info["matched_target_idx"]]
        transformation_matrix = joint_info["transformation_matrix"]

        source_points = []
        for v_idx in source_voxels:
            center = (np.array(v_idx) + 0.5) * voxel_size + shared_origin
            source_points.append(center)
        source_points_np = np.array(source_points)

        target_points = []
        for v_idx in target_voxels:
            center = (np.array(v_idx) + 0.5) * voxel_size + shared_origin
            target_points.append(center)
        target_points_np = np.array(target_points)

        if not np.allclose(transformation_matrix, np.eye(4)):
            source_points_homo = np.hstack([source_points_np, np.ones((len(source_points_np), 1))])
            transformed_points_homo = (transformation_matrix @ source_points_homo.T).T
            transformed_points = transformed_points_homo[:, :3]
        else:
            transformed_points = source_points_np

        all_points_with_colors.extend(source_points_np)
        all_colors.extend([joint_color] * len(source_points_np))
        
        all_points_with_colors.extend(transformed_points)
        all_colors.extend([joint_color] * len(transformed_points))
        
        all_points_with_colors.extend(target_points_np)
        all_colors.extend([joint_color] * len(target_points_np))

    combined_colored_pcd = o3d.geometry.PointCloud()
    combined_colored_pcd.points = o3d.utility.Vector3dVector(np.array(all_points_with_colors))
    combined_colored_pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))

    _voxel_vprint("Writing merged PLY...")

    all_geometries = [combined_source_pcd, combined_transformed_pcd, combined_target_pcd]
    if all_overlap_points:
        all_geometries.append(combined_overlap_pcd)

    merged_pcd = o3d.geometry.PointCloud()
    all_points = np.vstack([pcd.points for pcd in all_geometries])
    all_colors = np.vstack([pcd.colors for pcd in all_geometries])
    
    merged_pcd.points = o3d.utility.Vector3dVector(all_points)
    merged_pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    out_ply = os.path.join(save_dir, "combined_all_voxels.ply")
    o3d.io.write_point_cloud(out_ply, merged_pcd)
    print(f"[voxelize] combined_all_voxels.ply ({len(all_points)} points) -> {out_ply}")
    _voxel_vprint("Merge done (optional separate PLY writes are disabled).")
    _voxel_vprint(f"  source pts: {len(all_source_points)} (red)")
    _voxel_vprint(f"  transformed pts: {len(all_transformed_points)} (blue)")
    _voxel_vprint(f"  target pts: {len(all_target_points)} (green)")
    if all_overlap_points:
        _voxel_vprint(f"  overlap pts: {len(all_overlap_points)} (yellow)")
    _voxel_vprint(f"  colored merge pts: {len(all_points_with_colors)}")
    _voxel_vprint(f"  combined_all_voxels.ply: {len(all_points)} pts")
    _voxel_vprint(f"  dir: {save_dir}")

def _dilate_voxels(voxel_set, dilation_radius=1):
    """Morphological dilation on integer voxel coordinates (Chebyshev ball)."""
    if not voxel_set:
        return set()

    dilated_voxels = set(voxel_set)

    for voxel_coord in voxel_set:
        vx, vy, vz = voxel_coord

        for dx in range(-dilation_radius, dilation_radius + 1):
            for dy in range(-dilation_radius, dilation_radius + 1):
                for dz in range(-dilation_radius, dilation_radius + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue

                    dilated_coord = (vx + dx, vy + dy, vz + dz)
                    dilated_voxels.add(dilated_coord)

    return dilated_voxels


def _analyze_dilation_effect(original_intersection, dilated_intersection, voxel_set1, voxel_set2):
    """Stats: how many voxels move from dynamic to static after dilating the static intersection."""
    original_dynamic1 = len(voxel_set1 - original_intersection)
    original_dynamic2 = len(voxel_set2 - original_intersection)
    dilated_dynamic1 = len(voxel_set1 - dilated_intersection)
    dilated_dynamic2 = len(voxel_set2 - dilated_intersection)

    purified1 = original_dynamic1 - dilated_dynamic1
    purified2 = original_dynamic2 - dilated_dynamic2
    
    effect_info = {
        'original_intersection_count': len(original_intersection),
        'dilated_intersection_count': len(dilated_intersection),
        'expansion_count': len(dilated_intersection) - len(original_intersection),
        'original_dynamic1': original_dynamic1,
        'original_dynamic2': original_dynamic2,
        'dilated_dynamic1': dilated_dynamic1,
        'dilated_dynamic2': dilated_dynamic2,
        'purified1': purified1,
        'purified2': purified2,
        'total_purified': purified1 + purified2
    }
    
    return effect_info

def _find_connected_voxel_components(voxel_set, num_largest_components_to_keep=3, return_all_components=False):
    """26-connected components; returns list of voxel sets sorted by size (descending)."""
    if not voxel_set:
        return []

    visited = set()
    all_components = []

    for voxel_coord in voxel_set:
        if voxel_coord not in visited:
            current_component_voxels = set()
            q = deque()
            
            q.append(voxel_coord)
            visited.add(voxel_coord)
            current_component_voxels.add(voxel_coord)
            
            while q:
                vx, vy, vz = q.popleft()
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            
                            neighbor = (vx + dx, vy + dy, vz + dz)
                            
                            if neighbor in voxel_set and neighbor not in visited:
                                visited.add(neighbor)
                                q.append(neighbor)
                                current_component_voxels.add(neighbor)
            
            if current_component_voxels:
                all_components.append((len(current_component_voxels), current_component_voxels))
                
    all_components.sort(key=lambda item: item[0], reverse=True)

    sorted_components = [comp[1] for comp in all_components]
    
    return sorted_components

def extract_dynamic_joints(
    pcd_path1,
    pcd_path2,
    num_joints,
    voxel_size=0.03,
    save_dir=None,
    dilation_radius=1,
    verbose=False,
):
    """
    Extract per-joint voxel regions from two articulated point clouds (start/end).

    Args:
        pcd_path1: Path to first-state PLY.
        pcd_path2: Path to second-state PLY.
        num_joints: Expected number of dynamic parts (Top-K components per cloud).
        voxel_size: Voxel edge length in world units.
        save_dir: Output folder (e.g. point_cloud/iteration_xxx); writes joint_voxel_info.npy.
        dilation_radius: Dilate static intersection to strip boundary voxels from dynamic sets.
        verbose: Print diagnostics and show Open3D windows.

    Returns:
        joint_voxel_info dict (also saved as .npy when save_dir is set).
    """
    global _VOXEL_VERBOSE
    _prev_verbose = _VOXEL_VERBOSE
    _VOXEL_VERBOSE = verbose
    try:
        print(
            f"[voxelize] k={num_joints}, voxel_size={voxel_size}, "
            f"dilation_radius={dilation_radius}, out_dir={save_dir or '(none)'}"
        )
        _voxel_vprint("=" * 50)
        _voxel_vprint("Dynamic joint extraction...")
        _voxel_vprint(
            f"args: k={num_joints}, voxel_size={voxel_size}, dilation_radius={dilation_radius}"
        )
        _voxel_vprint("=" * 50)

        return _extract_dynamic_joints_impl(
            pcd_path1,
            pcd_path2,
            num_joints,
            voxel_size,
            save_dir,
            dilation_radius,
        )
    finally:
        _VOXEL_VERBOSE = _prev_verbose


def _extract_dynamic_joints_impl(
    pcd_path1, pcd_path2, num_joints, voxel_size, save_dir, dilation_radius
):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    pcd1_full = o3d.io.read_point_cloud(pcd_path1)
    pcd2_full = o3d.io.read_point_cloud(pcd_path2)
    pcd1_full.points = o3d.utility.Vector3dVector(pcd1_full.points)
    pcd2_full.points = o3d.utility.Vector3dVector(pcd2_full.points)

    bbox1 = pcd1_full.get_axis_aligned_bounding_box()
    bbox2 = pcd2_full.get_axis_aligned_bounding_box()

    min_bound = np.minimum(bbox1.min_bound, bbox2.min_bound)
    max_bound = np.maximum(bbox1.max_bound, bbox2.max_bound)
    combined_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    shared_origin = combined_bbox.min_bound

    points1_np = np.asarray(pcd1_full.points)
    points2_np = np.asarray(pcd2_full.points)

    voxel_indices1_all = np.floor((points1_np - shared_origin) / voxel_size).astype(int)
    voxel_indices2_all = np.floor((points2_np - shared_origin) / voxel_size).astype(int)

    voxel_to_points1 = {}
    for i, v_idx in enumerate(map(tuple, voxel_indices1_all)):
        if v_idx not in voxel_to_points1:
            voxel_to_points1[v_idx] = []
        voxel_to_points1[v_idx].append(i)

    voxel_to_points2 = {}
    for i, v_idx in enumerate(map(tuple, voxel_indices2_all)):
        if v_idx not in voxel_to_points2:
            voxel_to_points2[v_idx] = []
        voxel_to_points2[v_idx].append(i)

    voxel_set1 = set(voxel_to_points1.keys())
    voxel_set2 = set(voxel_to_points2.keys())

    intersection_voxels = voxel_set1 & voxel_set2

    if dilation_radius > 0:
        _voxel_vprint(f"\nDilate static intersection, radius={dilation_radius}")
        original_intersection = intersection_voxels.copy()
        dilated_intersection_voxels = _dilate_voxels(intersection_voxels, dilation_radius)
        _voxel_vprint(f"intersection voxels before: {len(intersection_voxels)}")
        _voxel_vprint(f"intersection voxels after: {len(dilated_intersection_voxels)}")

        effect_info = _analyze_dilation_effect(
            original_intersection, dilated_intersection_voxels, voxel_set1, voxel_set2
        )
        _voxel_vprint("Dilation stats:")
        _voxel_vprint(f"  intersection growth: {effect_info['expansion_count']}")
        _voxel_vprint(
            f"  source purified: {effect_info['purified1']} "
            f"({effect_info['original_dynamic1']} -> {effect_info['dilated_dynamic1']})"
        )
        _voxel_vprint(
            f"  target purified: {effect_info['purified2']} "
            f"({effect_info['original_dynamic2']} -> {effect_info['dilated_dynamic2']})"
        )
        _voxel_vprint(f"  total purified: {effect_info['total_purified']}")

        intersection_voxels = dilated_intersection_voxels

    unique_voxels1_initial = voxel_set1 - intersection_voxels
    unique_voxels2_initial = voxel_set2 - intersection_voxels

    _voxel_vprint("=" * 50)
    _voxel_vprint("Voxel counts:")
    _voxel_vprint(f"source dynamic voxels: {len(unique_voxels1_initial)}")
    _voxel_vprint(f"target dynamic voxels: {len(unique_voxels2_initial)}")

    _voxel_vprint("\nTop-K connected components...")

    source_components = _find_connected_voxel_components(
        unique_voxels1_initial, num_joints, return_all_components=True
    )
    target_components = _find_connected_voxel_components(
        unique_voxels2_initial, num_joints, return_all_components=True
    )

    _voxel_vprint(f"source components found: {len(source_components)}")
    _voxel_vprint(f"target components found: {len(target_components)}")

    source_components = source_components[:num_joints]
    target_components = target_components[:num_joints]

    _voxel_vprint(f"source components kept: {len(source_components)}")
    _voxel_vprint(f"target components kept: {len(target_components)}")

    unmatched_source_voxels = set()
    for comp in source_components:
        unmatched_source_voxels.update(comp)
    
    unmatched_target_voxels = set()
    for comp in target_components:
        unmatched_target_voxels.update(comp)

    _voxel_vprint("=" * 50)
    _voxel_vprint("Segmentation (Top-K):")
    _voxel_vprint(f"source top-{num_joints} components:")
    for i, comp in enumerate(source_components):
        comp_points_count = sum(len(voxel_to_points1[v_idx]) for v_idx in comp)
        _voxel_vprint(f"  comp {i+1}: {len(comp)} voxels, {comp_points_count} points")

    _voxel_vprint(f"target top-{num_joints} components:")
    for i, comp in enumerate(target_components):
        comp_points_count = sum(len(voxel_to_points2[v_idx]) for v_idx in comp)
        _voxel_vprint(f"  comp {i+1}: {len(comp)} voxels, {comp_points_count} points")

    _voxel_vprint("\nVoxel occupancy (points per voxel):")
    all_voxel_densities = [len(voxel_to_points1[v_idx]) for v_idx in unmatched_source_voxels]
    if all_voxel_densities:
        _voxel_vprint(
            f"source: min={min(all_voxel_densities)}, max={max(all_voxel_densities)}, "
            f"mean={np.mean(all_voxel_densities):.2f}"
        )
    else:
        _voxel_vprint("source: no dynamic voxels")

    all_voxel_densities = [len(voxel_to_points2[v_idx]) for v_idx in unmatched_target_voxels]
    if all_voxel_densities:
        _voxel_vprint(
            f"target: min={min(all_voxel_densities)}, max={max(all_voxel_densities)}, "
            f"mean={np.mean(all_voxel_densities):.2f}"
        )
    else:
        _voxel_vprint("target: no dynamic voxels")

    joint_voxel_info = {
        'voxel_size': voxel_size,
        'shared_origin': shared_origin,
        'dilation_radius': dilation_radius,
        'source_joints': [],
        'target_joints': [],
        'voxel_to_points1': voxel_to_points1,
        'voxel_to_points2': voxel_to_points2,
        'points1_np': points1_np,
        'points2_np': points2_np
    }
    
    has_target_components = len(target_components) > 0 and any(len(comp) > 0 for comp in target_components)

    if not has_target_components:
        print("[voxelize] warning: no target dynamic voxels; saving source-only joints (no match)")
        _voxel_vprint("\n" + "=" * 50)
        _voxel_vprint("No target dynamic voxels; skip matching, keep source components only")
        _voxel_vprint("=" * 50)

        for i, source_comp in enumerate(source_components):
            joint_info = {
                "joint_id": i,
                "source_voxels": list(source_comp),
                "target_voxels": [],
                "source_point_indices": [],
                "target_point_indices": [],
                "transformation_matrix": np.eye(4),
                "matched_target_idx": -1,
                "transform_valid": True,
                "transform_error_info": "no target voxels; identity",
                "overlap_count": 0,
                "overlap_ratio": 0.0,
            }

            for v_idx in source_comp:
                if v_idx in voxel_to_points1:
                    joint_info["source_point_indices"].extend(voxel_to_points1[v_idx])

            joint_voxel_info["source_joints"].append(joint_info)
            joint_voxel_info["target_joints"].append(joint_info)

            _voxel_vprint(
                f"joint {i+1}: {len(source_comp)} src voxels, "
                f"{len(joint_info['source_point_indices'])} src pts, no target"
            )

        _voxel_vprint("\nSkip matching / transforms; continue to visualization...")
    else:
        _voxel_vprint("\n" + "=" * 50)
        _voxel_vprint("Component matching and transforms...")
        matched_pairs = match_voxel_components(source_components, target_components, voxel_size, shared_origin)

        for i, (source_comp, target_comp) in enumerate(zip(source_components, target_components)):
            joint_info = {
                "joint_id": i,
                "source_voxels": list(source_comp),
                "target_voxels": list(target_comp),
                "source_point_indices": [],
                "target_point_indices": [],
                "transformation_matrix": np.eye(4),
                "matched_target_idx": -1,
            }

            for v_idx in source_comp:
                if v_idx in voxel_to_points1:
                    joint_info["source_point_indices"].extend(voxel_to_points1[v_idx])

            for v_idx in target_comp:
                if v_idx in voxel_to_points2:
                    joint_info["target_point_indices"].extend(voxel_to_points2[v_idx])

            matched_target_idx = -1
            overlap_count = 0
            overlap_ratio = 0.0
            
            for source_idx, target_idx in matched_pairs:
                if source_idx == i:
                    matched_target_idx = target_idx
                    break
            
            if matched_target_idx != -1:
                _voxel_vprint(f"\nJoint {i+1} match refine...")
                matched_target_comp = target_components[matched_target_idx]
                s_ratio, _, _ = component_aspect_ratio(source_comp, voxel_size, shared_origin)
                t_ratio, _, _ = component_aspect_ratio(matched_target_comp, voxel_size, shared_origin)
                if s_ratio < ASPECT_RATIO_THRESHOLD or t_ratio < ASPECT_RATIO_THRESHOLD:
                    joint_info["transformation_matrix"] = np.eye(4)
                    joint_info["transform_valid"] = False
                    joint_info["transform_error_info"] = (
                        f"low aspect (src {s_ratio:.2f}, tgt {t_ratio:.2f} < {ASPECT_RATIO_THRESHOLD:.2f}); skip coarse pose"
                    )
                    _voxel_vprint(
                        f"  joint {i+1} low aspect (src {s_ratio:.2f}, tgt {t_ratio:.2f}), skip coarse pose"
                    )
                    overlap_count, overlap_ratio = 0, 0.0
                else:

                    source_plane, target_plane = fit_planes_from_matched_voxels(
                        source_comp, matched_target_comp, voxel_size, shared_origin
                    )

                    if source_plane is not None and target_plane is not None:
                        transformation_matrix = calculate_transform_from_planes(
                            source_plane, target_plane, source_comp, matched_target_comp, voxel_size, shared_origin
                        )

                        overlap_count, overlap_ratio = calculate_voxel_overlap_after_transform(
                            source_comp, matched_target_comp, transformation_matrix, voxel_size, shared_origin
                        )

                        is_valid, error_info = validate_transformation_matrix(
                            transformation_matrix, source_comp, matched_target_comp, voxel_size, shared_origin
                        )

                        if overlap_ratio < 0.1:
                            joint_info["transformation_matrix"] = np.eye(4)
                            joint_info["transform_valid"] = False
                            joint_info["transform_error_info"] = (
                                f"overlap too low ({overlap_ratio:.3f} < 0.1); identity"
                            )
                            _voxel_vprint(
                                f"joint {i+1} overlap too low ({overlap_ratio:.3f} < 0.1); identity"
                            )
                        else:
                            joint_info["transformation_matrix"] = transformation_matrix
                            joint_info["transform_valid"] = is_valid
                            joint_info["transform_error_info"] = error_info
                            _voxel_vprint(
                                f"joint {i+1} ok - overlap {overlap_count}, ratio {overlap_ratio:.3f}, {error_info}"
                            )
                    else:
                        joint_info["transformation_matrix"] = np.eye(4)
                        joint_info["transform_valid"] = False
                        joint_info["transform_error_info"] = "plane fit failed"
                        _voxel_vprint(f"joint {i+1} plane fit failed; identity")
            else:
                joint_info["transformation_matrix"] = np.eye(4)
                joint_info["transform_valid"] = True
                joint_info["transform_error_info"] = "unmatched; identity"
                _voxel_vprint(f"joint {i+1}: no matched target; identity")

            joint_info["matched_target_idx"] = matched_target_idx
            joint_info["overlap_count"] = overlap_count
            joint_info["overlap_ratio"] = overlap_ratio

            joint_voxel_info["source_joints"].append(joint_info)
            joint_voxel_info["target_joints"].append(joint_info)

            _voxel_vprint(
                f"joint {i+1}: src vox {len(source_comp)}, src pts {len(joint_info['source_point_indices'])}, "
                f"tgt vox {len(target_comp)}, tgt pts {len(joint_info['target_point_indices'])}, "
                f"matched_idx {matched_target_idx}"
            )

        print_transformation_summary(joint_voxel_info)

        if _VOXEL_VERBOSE:
            visualize_transformed_matching(
                joint_voxel_info,
                source_components,
                target_components,
                voxel_size,
                shared_origin,
                save_dir,
            )
        
        save_combined_matched_voxels(
            joint_voxel_info,
            source_components,
            target_components,
            voxel_size,
            shared_origin,
            save_dir
        )
    
    _print_joint_brief_summary(joint_voxel_info)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        voxel_info_path = os.path.join(save_dir, "joint_voxel_info.npy")
        np.save(voxel_info_path, joint_voxel_info)
        print(f"[voxelize] joint_voxel_info.npy -> {voxel_info_path}")

    SHOW_STATIC_BACKGROUND = False

    debug_visualize_components(
        source_components,
        pcd1_full,
        voxel_to_points1,
        points1_np,
        "Top-K segmentation (source)",
        show_static_background=SHOW_STATIC_BACKGROUND,
        save_dir=save_dir,
    )
    return joint_voxel_info

def debug_visualize_components(
    source_components,
    pcd1_full,
    voxel_to_points1,
    points1_np,
    window_title="Debug Components",
    show_static_background=False,
    save_dir=None,
    min_voxels_per_component=1,
):
    """Color each connected component; optional full cloud as gray background."""
    _voxel_vprint(f"debug viz: {window_title}")

    colors = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [1, 0, 0.5], [0.5, 1, 0], [0, 0.5, 1]
    ]
    
    geometries = []

    if show_static_background:
        static_pcd = o3d.geometry.PointCloud()
        static_pcd.points = pcd1_full.points
        static_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.5, 0.5, 0.5], (len(pcd1_full.points), 1)))
        geometries.append(static_pcd)
        _voxel_vprint("static background (gray)")

    all_dynamic_voxels = set()
    for comp in source_components:
        all_dynamic_voxels.update(comp)
    
    _voxel_vprint(f"dynamic voxels: {len(all_dynamic_voxels)}")

    total_dynamic_points = 0
    for v_idx in all_dynamic_voxels:
        if v_idx in voxel_to_points1:
            total_dynamic_points += len(voxel_to_points1[v_idx])
    _voxel_vprint(f"dynamic points: {total_dynamic_points}")

    has_valid_components = False
    for i, comp in enumerate(source_components):
        if len(comp) >= min_voxels_per_component:
            has_valid_components = True
            break
    
    for i, comp in enumerate(source_components):
        comp_points_indices = []
        for v_idx in comp:
            if v_idx in voxel_to_points1:
                comp_points_indices.extend(voxel_to_points1[v_idx])
        
        if comp_points_indices:
            comp_pcd = o3d.geometry.PointCloud()
            comp_pcd.points = o3d.utility.Vector3dVector(points1_np[comp_points_indices])

            color = colors[i % len(colors)]
            comp_pcd.paint_uniform_color(color)

            geometries.append(comp_pcd)
            _voxel_vprint(f"  comp {i+1}: {len(comp)} voxels, {len(comp_points_indices)} pts, color {color}")

    if geometries:
        _voxel_vprint(f"{len(geometries)} geometries ({len(source_components)} dynamic comps)")
    else:
        _voxel_vprint("warning: no dynamic components to visualize")

    if save_dir:
        if geometries and has_valid_components:
            merged_pcd = o3d.geometry.PointCloud()
            merged_pcd.points = o3d.utility.Vector3dVector(np.concatenate([pcd.points for pcd in geometries]))
            merged_pcd.colors = o3d.utility.Vector3dVector(np.concatenate([pcd.colors for pcd in geometries]))
            dyn_path = os.path.join(save_dir, "dynamic_components.ply")
            o3d.io.write_point_cloud(dyn_path, merged_pcd)
            print(f"[voxelize] dynamic_components.ply -> {dyn_path}")
        elif geometries and not has_valid_components:
            _voxel_vprint(
                f"skip dynamic_components.ply: no comp >= {min_voxels_per_component} voxels"
            )
    else:
        _voxel_vprint("no save_dir; skip dynamic_components.ply")
    
    if geometries and _VOXEL_VERBOSE:
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_title
        )

def debug_visualize_comparison(source_components, pcd1_full, voxel_to_points1, points1_np, window_title="Compare"):
    """Show full cloud (gray) plus colored per-component clouds."""
    print(f"comparison viz: {window_title}")

    colors = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [1, 0, 0.5], [0.5, 1, 0], [0, 0.5, 1]
    ]
    
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = pcd1_full.points
    original_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.7, 0.7, 0.7], (len(pcd1_full.points), 1)))
    
    segmented_geometries = []
    
    for i, comp in enumerate(source_components):
        comp_points_indices = []
        for v_idx in comp:
            if v_idx in voxel_to_points1:
                comp_points_indices.extend(voxel_to_points1[v_idx])
        
        if comp_points_indices:
            comp_pcd = o3d.geometry.PointCloud()
            comp_pcd.points = o3d.utility.Vector3dVector(points1_np[comp_points_indices])

            color = colors[i % len(colors)]
            comp_pcd.paint_uniform_color(color)

            segmented_geometries.append(comp_pcd)
    
    all_geometries = [original_pcd] + segmented_geometries
    print(f"show: full cloud + {len(segmented_geometries)} segmented comps")
    
    o3d.visualization.draw_geometries(
        all_geometries,
        window_name=window_title
    )

def create_joint_masks_from_voxel_info(voxel_info, point_cloud_points):
    """Build per-joint boolean masks (N,) from saved joint_voxel_info."""
    voxel_size = voxel_info["voxel_size"]
    shared_origin = voxel_info["shared_origin"]

    voxel_indices = np.floor((point_cloud_points - shared_origin) / voxel_size).astype(int)
    voxel_coords = [tuple(v_idx) for v_idx in voxel_indices]
    
    joint_masks = []
    
    for joint_info in voxel_info["source_joints"]:
        joint_mask = np.zeros(len(point_cloud_points), dtype=bool)

        for v_idx in joint_info["source_voxels"]:
            for i, v_coord in enumerate(voxel_coords):
                if v_coord == v_idx:
                    joint_mask[i] = True

        joint_masks.append(joint_mask)
        print(f"joint {joint_info['joint_id']+1}: {np.sum(joint_mask)} points marked")
    
    return joint_masks

def get_joint_point_indices_from_voxel_info(voxel_info, joint_id):
    """Return (source_point_indices, target_point_indices) for joint_id."""
    if joint_id >= len(voxel_info["source_joints"]):
        raise ValueError(
            f"joint_id {joint_id} out of range (max {len(voxel_info['source_joints'])-1})"
        )

    joint_info = voxel_info["source_joints"][joint_id]
    return joint_info["source_point_indices"], joint_info["target_point_indices"]


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_json(repo_root, rel_path):
    path = os.path.join(repo_root, rel_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _joint_types_string(joint_types_pcd, dataset, subset, scene):
    try:
        return joint_types_pcd[dataset][subset][scene]
    except KeyError as e:
        raise KeyError(
            f"scene {dataset}/{subset}/{scene} not in joint_types_pcd.json"
        ) from e


def _num_joints_from_types(joint_types_str):
    return len(joint_types_str.split(","))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Batch voxelize: read larger_motion from point_cloud/iteration_*/larger_motion_state.txt; "
            "K from joint_types_pcd.json (comma-separated types). Calls extract_dynamic_joints."
        )
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--scenes", type=str, nargs="+", required=True, help="One or more scene names")
    parser.add_argument(
        "--iteration",
        type=int,
        required=True,
        help="Coarse iteration folder point_cloud/iteration_<N>",
    )
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel edge length (default 0.01)")
    parser.add_argument(
        "--dilation_radius",
        type=int,
        default=1,
        help="Dilate static intersection (default 1; 0 disables)",
    )
    parser.add_argument(
        "--coarse_name",
        type=str,
        default="coarse_gs",
        help="Subdir under outputs/<dataset>/<subset>/<scene> (train_coarse model name)",
    )
    parser.add_argument(
        "--joint_types_pcd_json",
        type=str,
        default="arguments/joint_types_pcd.json",
        help="JSON path for joint type string (K = number of comma-separated entries)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logs and Open3D GUI",
    )
    args = parser.parse_args()

    root = _repo_root()
    joint_types_pcd = _load_json(root, args.joint_types_pcd_json)

    for scene in args.scenes:
        try:
            jt_str = _joint_types_string(joint_types_pcd, args.dataset, args.subset, scene)
        except KeyError as e:
            print(f"skip {scene}: {e}")
            continue

        k_joints = _num_joints_from_types(jt_str)
        coarse_dir = os.path.join(root, "outputs", args.dataset, args.subset, scene, args.coarse_name)
        ply_dir = os.path.join(coarse_dir, "point_cloud", f"iteration_{args.iteration}")
        lm_path = os.path.join(ply_dir, "larger_motion_state.txt")
        if not os.path.isfile(lm_path):
            print(f"skip {scene}: missing larger_motion_state.txt\n  {lm_path}")
            continue
        with open(lm_path, "r", encoding="utf-8") as f:
            lm = int(f.read().strip())
        pcd_path1 = os.path.join(ply_dir, f"point_cloud_{lm}.ply")
        pcd_path2 = os.path.join(ply_dir, f"point_cloud_{int(not lm)}.ply")

        if not os.path.isfile(pcd_path1) or not os.path.isfile(pcd_path2):
            print(f"skip {scene}: missing point clouds\n  {pcd_path1}\n  {pcd_path2}")
            continue

        print(
            f"\n>>> {args.dataset}/{args.subset}/{scene} | K={k_joints} | larger_motion={lm} | "
            f"iter={args.iteration} | voxel={args.voxel_size} | dilate={args.dilation_radius}"
        )
        print(f"    joint_types: {jt_str}")

        joint_voxel_info = extract_dynamic_joints(
            pcd_path1=pcd_path1,
            pcd_path2=pcd_path2,
            num_joints=k_joints,
            voxel_size=args.voxel_size,
            save_dir=ply_dir,
            dilation_radius=args.dilation_radius,
            verbose=args.verbose,
        )

        if joint_voxel_info:
            print(f"[voxelize] done {scene}, output: {ply_dir}")