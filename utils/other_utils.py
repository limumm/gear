import torch
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.optimize import linear_sum_assignment
from utils.pointnet2_utils import farthest_point_sample, index_points
import os

def match_pcd(pc0, pc1, N=5000):
    """
    Input:
        pc0, pc1: tensor [1, N0, 3], [1, N1, 3]
        N: downsample number
    Return:
        idx_s, idx_e: [N], [N]
    """
    # Downsample with farthest point sampling
    # s: start, e: end
    num_fps = min(pc0.shape[1], pc1.shape[1], N)
    s_idx = farthest_point_sample(pc0, num_fps)
    pc_s = index_points(pc0, s_idx)
    e_idx = farthest_point_sample(pc1, num_fps)
    pc_e = index_points(pc1, e_idx)

    # Matching
    with torch.no_grad():
        cost = torch.cdist(pc_s, pc_e).cpu().numpy() # shape: [1, num_fps, num_fps]
    idx_s, idx_e = linear_sum_assignment(cost.squeeze())
    idx_s, idx_e = s_idx[0].cpu().numpy()[idx_s], e_idx[0].cpu().numpy()[idx_e]
    return idx_s, idx_e

def get_larger_motion_state(path, num_slots, visualize=False, voxel_size=0.02, compute_inverse_transform=False):
    xyzs = []
    pcds_full = []
    ply_data = []
    
    for state in (0, 1):
        plypath = path.replace('point_cloud.ply', f'point_cloud_{state}.ply')
        if not os.path.exists(plypath):
            print(f"Error: File not found at {plypath}")
            return None, None
        plydata = PlyData.read(plypath)
        ply_data.append(plydata)
        
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        xyzs.append(xyz)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcds_full.append(pcd)

    print("Determining source and target states based on average point distance...")
    dists0_to_1 = np.asarray(pcds_full[0].compute_point_cloud_distance(pcds_full[1]))
    dists1_to_0 = np.asarray(pcds_full[1].compute_point_cloud_distance(pcds_full[0]))
    avg_dist0_to_1 = np.mean(dists0_to_1)
    avg_dist1_to_0 = np.mean(dists1_to_0)
    print(f"  - Avg distance from state 0 to 1: {avg_dist0_to_1:.4f}")
    print(f"  - Avg distance from state 1 to 0: {avg_dist1_to_0:.4f}")

<<<<<<< HEAD
    if avg_dist0_to_1 > avg_dist1_to_0:
        source_state_idx, target_state_idx = 0, 1
=======
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (cano_gs.max_sh_degree + 1) ** 2 - 1))
        features_extras.append(features_extra)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scale[:, idx] = np.asarray(plydata.elements[0][attr_name])
        scales.append(scale)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rot[:, idx] = np.asarray(plydata.elements[0][attr_name])
        rots.append(rot)

        fea_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("fea")]
        feat = np.zeros((xyz.shape[0], cano_gs.fea_dim))
        for idx, attr_name in enumerate(fea_names):
            feat[:, idx] = np.asarray(plydata.elements[0][attr_name])
        feats.append(feat)

    pc0, pc1 = torch.tensor(xyzs[0])[None].cuda(), torch.tensor(xyzs[1])[None].cuda()
    idx = match_pcd(pc0, pc1) # idx: [idx_start, idx_end]

    cd, _ = chamfer_distance(pc0, pc1, batch_reduction=None, point_reduction=None) # cd: [cd_start2end, cd_end2start]
    
    larger_motion_state = 0 if cd[0].mean().item() > cd[1].mean().item() else 1
    print("Larger motion state: ", larger_motion_state)

    threshould = [cano_gs.dynamic_threshold_ratio * cd[0].max().item(), cano_gs.dynamic_threshold_ratio * cd[1].max().item()]
    mask_static = [(cd[i].squeeze() < threshould[i]).cpu().numpy() for i in range(2)]
    mask_dynamic = [~mask_static[i] for i in range(2)]

    s = larger_motion_state
    xyz = np.concatenate([xyzs[s][mask_static[s]], (xyzs[0][idx[0]] + xyzs[1][idx[1]]) * 0.5])
    opacities = np.concatenate([opacities[s][mask_static[s]], (opacities[0][idx[0]] + opacities[1][idx[1]]) * 0.5])
    features_dcs = np.concatenate([features_dcs[s][mask_static[s]], (features_dcs[0][idx[0]] + features_dcs[1][idx[1]]) * 0.5])
    features_extras = np.concatenate([features_extras[s][mask_static[s]], (features_extras[0][idx[0]] + features_extras[1][idx[1]]) * 0.5])
    scales = np.concatenate([scales[s][mask_static[s]], (scales[0][idx[0]] + scales[1][idx[1]]) * 0.5])
    rots = np.concatenate([rots[s][mask_static[s]], (rots[0][idx[0]] + rots[1][idx[1]]) * 0.5])
    feats = np.concatenate([feats[s][mask_static[s]], (feats[0][idx[0]] + feats[1][idx[1]]) * 0.5])

    cano_gs._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    cano_gs._features_dc = nn.Parameter(torch.tensor(features_dcs, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    cano_gs._features_rest = nn.Parameter(torch.tensor(features_extras, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    cano_gs._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    cano_gs._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    cano_gs._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    if cano_gs.fea_dim > 0:
        cano_gs.feature = nn.Parameter(torch.tensor(feats, dtype=torch.float, device="cuda").requires_grad_(True))

    cano_gs.max_radii2D = torch.zeros((cano_gs.get_xyz.shape[0]), device="cuda")
    cano_gs.active_sh_degree = cano_gs.max_sh_degree
    cano_gs.save_ply(path)

    if num_slots > 3 or 'real' in path: # larger threshold for complex or real wolrd multi-part objects
        ratio = 0.05
        threshould = [ratio * cd[0].max().item(), ratio * cd[1].max().item()]
        mask_static = [(cd[i].squeeze() < threshould[i]).cpu().numpy() for i in range(2)]
        mask_dynamic = [~mask_static[i] for i in range(2)]
    np.save(path.replace('point_cloud.ply', 'xyz_static.npy'), xyzs[s][mask_static[s]])
    np.save(path.replace('point_cloud.ply', 'xyz_dynamic.npy'), xyzs[s][mask_dynamic[s]])
    np.save(path.replace('point_cloud.ply', 'xyz_static_0.npy'), xyzs[0][mask_static[0]])
    np.save(path.replace('point_cloud.ply', 'xyz_dynamic_0.npy'), xyzs[0][mask_dynamic[0]])
    np.save(path.replace('point_cloud.ply', 'xyz_static_1.npy'), xyzs[1][mask_static[1]])
    np.save(path.replace('point_cloud.ply', 'xyz_dynamic_1.npy'), xyzs[1][mask_dynamic[1]])
    if visualize:
        import seaborn as sns
        pallete = np.array(sns.color_palette("hls", 2))
        point_cloud = o3d.geometry.PointCloud()
        x_s = xyzs[s][mask_static[s]]
        x_matched = (xyzs[0][idx[0]] + xyzs[1][idx[1]]) * 0.5
        x = np.concatenate([x_s, x_matched])
        color_s = np.tile(pallete[0], (x_s.shape[0], 1))
        color_m = np.tile(pallete[1], (x_matched.shape[0], 1))
        color = np.concatenate([color_s, color_m])
        color = np.clip(color, 0, 1)
        point_cloud.points = o3d.utility.Vector3dVector(x)
        point_cloud.colors = o3d.utility.Vector3dVector(color.astype(np.float64))
        o3d.visualization.draw_geometries([point_cloud])
    return larger_motion_state


def cal_cluster_centers(cano_path, num_slots, visualize=False):
    xyz_static = np.load(cano_path.replace('point_cloud.ply', 'xyz_static.npy'))
    xyz_dynamic = np.load(cano_path.replace('point_cloud.ply', 'xyz_dynamic.npy'))
    print("Finding centers by Spectral Clustering")
    if num_slots > 2:
        cluster = SpectralClustering(num_slots - 1, assign_labels='discretize', random_state=0)
        labels = cluster.fit_predict(xyz_dynamic)
        center_dynamic = np.array([xyz_dynamic[labels == i].mean(0) for i in range(num_slots - 1)])
        labels = np.concatenate([np.zeros(xyz_static.shape[0]), labels + 1])
        center = np.concatenate([xyz_static.mean(0, keepdims=True), center_dynamic])
>>>>>>> main
    else:
        source_state_idx, target_state_idx = 1, 0
    
    print(f"Decision: State {source_state_idx} is Source, State {target_state_idx} is Target.")

    return source_state_idx