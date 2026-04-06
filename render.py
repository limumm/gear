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

import copy
import json
import os
from argparse import ArgumentParser
from os import makedirs

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torchvision
from pytorch_lightning import seed_everything
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import DeformModel, Scene
from scene.cameras import Camera
from utils.general_utils import safe_state
from utils.geo_utils import find_biggest_cluster
from utils.mesh_utils import GaussianExtractor
from utils.metrics import *

def generate_camera_poses(r=3, N=30):
    """
    Generate camera poses around the scene center on a circle.

    Parameters:
    - r: Radius of the circle.
    - theta: Elevation angle in degrees.
    - num_samples: Number of samples (camera positions) to generate.

    Returns:
    - poses: A list of camera poses (4x4 transformation matrices).
    """
    poses = []
    for i, theta in enumerate(range(-85, 85, 10)):
        theta_rad = np.deg2rad(theta)

        # Generate azimuth angles evenly spaced around the circle
        num_samples = 25 - abs(theta) // 4
        azimuths = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)

        for azimuth in azimuths:
            # Convert spherical coordinates to Cartesian coordinates
            x = r * np.cos(azimuth) * np.cos(theta_rad)
            y = r * np.sin(azimuth) * np.cos(theta_rad)
            z = r * np.sin(theta_rad)

            # Camera position
            position = np.array([x, y, z])

            # Compute the forward direction (pointing towards the origin)
            forward = position / np.linalg.norm(position)

            # Compute the right and up vectors for the camera coordinate system
            up = np.array([0, 0, 1])
            if np.allclose(forward, up) or np.allclose(forward, -up):
                up = np.array([0, 1, 0])
            right = np.cross(up, forward)
            up = np.cross(forward, right)

            # Normalize the vectors
            right /= np.linalg.norm(right)
            up /= np.linalg.norm(up)

            # Construct the rotation matrix
            rotation_matrix = np.vstack([right, up, forward]).T

            # Construct the transformation matrix (4x4)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = position

            poses.append(transformation_matrix)

    return poses


def generate(views):
    new_views = []
    poses = generate_camera_poses(2)
    for i, pose in enumerate(poses):
        view = copy.deepcopy(views[0])
        view.fid = np.random.randint(2, size=1).item()
        view.gt_alpha_mask = None
        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        view.reset_extrinsic(R, T)
        new_views.append(view)
    return new_views


def get_rotation_axis_angle(k, theta):
    '''
    Rodrigues' rotation formula
    args:
    * k: direction unit vector of the axis to rotate about
    * theta: the (radian) angle to rotate with
    return:
    * 3x3 rotation matrix
    '''
    if np.linalg.norm(k) == 0.:
        return np.eye(3)
    k = k / np.linalg.norm(k)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3))
    R[0, 0] = cos + (kx**2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky**2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz**2) * (1 - cos)
    return R


def save_axis_mesh(k, center, filepath):
    """Export joint axis as arrow mesh (rotate joints only in this path)."""
    axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.7, cone_height=0.04)
    arrow = np.array([0., 0., 1.], dtype=np.float32)
    n = np.cross(arrow, k)
    rad = np.arccos(np.dot(arrow, k))
    R_arrow = get_rotation_axis_angle(n, rad)
    axis.rotate(R_arrow, center=(0, 0, 0))
    axis.translate(center[:3])
    o3d.io.write_triangle_mesh(filepath, axis)


joint_type_dict = {
    'r': 'hinge',
    'p': 'slider',
}


def export_joint_info_json(pred_joint_list, mesh_files, exp_dir):
    meta_info = []
    for i, joint_info in enumerate(pred_joint_list):
        if i == 0:
            entry = {
            "id": i,
            "parent": -1,
            "name": "root",
            "joint": 'heavy',
            "jointData": {},
            "visuals": [
                mesh_files[i]
            ]
        }
        else:
            entry = {
                "id": i,
                "parent": 0,
                "name": f"joint_{i}",
                "joint": joint_type_dict[joint_info['type']],
                "jointData": {
                    "axis": {
                        "origin": joint_info['axis_position'].tolist(),
                        "direction": joint_info['axis_direction'].tolist()
                    },
                    "limit": {
                    }
                },
                "visuals": [
                    mesh_files[i]
                ]
            }
        meta_info.append(entry)
    with open(os.path.join(exp_dir, 'joint_info.json'), 'w') as f:
        json.dump(meta_info, f, indent=4)


def render_set(args, load2gpt_on_the_fly, name, iteration, views: list[Camera], gaussians, pipeline, background, deform: DeformModel, eval_app=False, visualize=False):
    model_path = args.model_path
    larger_motion_state = args.larger_motion_state
    reverse = False if args.larger_motion_state == 0 else True
    print(f"larger_motion_state: {larger_motion_state}")
    
    print(f"reverse: {reverse}")
    pred_joint_types = deform.deform.joint_types[1:]
    d_values_list = deform.step(gaussians, is_training=False)
    num_d_joints = len(pred_joint_types)
    pred_joint_list = deform.deform.get_joint_param()
    
    save_dir = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(save_dir, exist_ok=True)
    mesh_path = os.path.join(model_path, name, "ours_{}".format(iteration), "meshes")
    makedirs(mesh_path, exist_ok=True)

    mask_hard = gaussians.mask_logits.argmax(-1)
    for i, joint_info in enumerate(pred_joint_list):
        pos = joint_info['axis_position']
        center = gaussians.get_xyz[mask_hard == i].mean(0).cpu().numpy()
        if pred_joint_types[i] == 'p':
            pos = center
        else:
            pos += joint_info['axis_direction'] * np.dot(joint_info['axis_direction'], center - pos)
        save_axis_mesh(joint_info['axis_direction'], pos, 
                       f'{mesh_path}/axis_{i}_{pred_joint_types[i]}.ply')
    try:
        gt_info_list = read_gt(os.path.expanduser(f'{args.source_path}/gt/trans.json'))
        results_list, real_perm = eval_axis_and_state_all(pred_joint_list, pred_joint_types, gt_info_list, print_perm=True, reverse=reverse)
        print(results_list)
    except Exception:
        print("No ground truth")
    
    PSNR, SSIM, LPIPS = -1, -1, -1
    for mask_id in range(-1, num_d_joints + 1):
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders", "{}".format(mask_id))
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt", "{}".format(mask_id))
        depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth", "{}".format(mask_id))

        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        makedirs(depth_path, exist_ok=True)

        rgbs_start, depths_start = [], []
        rgbs_end, depths_end = [], []
        alphas_start, alphas_end = [], []
        gt_rgbs_start, gt_rgbs_end = [], []
        views_start, views_end = [], []
        
        if mask_id > 0 and "real_" in args.source_path:
            x = gaussians.get_xyz
            mask_part = d_values_list[0]["mask"] == mask_id
            _, mask_cluster = find_biggest_cluster(x[mask_part].cpu().numpy(), eps=0.05, min_samples=2)
            print("Removed {} noise Gaussians".format((mask_cluster == False).sum()))
            keep_mask = torch.ones(len(x), dtype=torch.bool).cuda()
            keep_mask[mask_part] = torch.tensor(mask_cluster, dtype=torch.bool).cuda()
        else:
            keep_mask = None

        for idx, view in enumerate(tqdm(views)):
            if load2gpt_on_the_fly:
                view.load2device()
            gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
            d_values = d_values_list[0]
            mask = d_values["mask"]

            is_start = view.fid < 0.5
            use_deformation = (is_start and larger_motion_state == 1) or (
                not is_start and larger_motion_state == 0
            )

            vis_mask = mask == mask_id if mask_id != -1 else None
            if use_deformation:
                if keep_mask is not None and vis_mask is not None:
                    vis_mask = vis_mask & keep_mask
                d_xyz, d_rotation = d_values["d_xyz"], d_values["d_rotation"]
                results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, vis_mask=vis_mask)
            else:
                results = render(view, gaussians, pipeline, background, d_xyz=None, d_rot=None, vis_mask=vis_mask)

            if is_start:
                gt_rgbs_start.append(gt_image)
                views_start.append(view)
                rgbs_start.append(torch.clamp(results["render"], 0.0, 1.0).cpu())
                alphas_start.append(torch.clamp(results["alpha"], 0.0, 1.0).cpu())
                depths_start.append(results["depth"].cpu())
            else:
                gt_rgbs_end.append(gt_image)
                views_end.append(view)
                rgbs_end.append(torch.clamp(results["render"], 0.0, 1.0).cpu())
                alphas_end.append(torch.clamp(results["alpha"], 0.0, 1.0).cpu())
                depths_end.append(results["depth"].cpu())

            del results
            if load2gpt_on_the_fly:
                torch.cuda.empty_cache()
        if name == 'train':
            gsExtractor_start = GaussianExtractor(
                views_start, rgbs_start, depths_start, alphas_start, gaussians, render, pipeline, background
            )
            mesh_start = gsExtractor_start.extract_mesh()
            del gsExtractor_start
            torch.cuda.empty_cache()

            gsExtractor_end = GaussianExtractor(
                views_end, rgbs_end, depths_end, alphas_end, gaussians, render, pipeline, background
            )
            mesh_end = gsExtractor_end.extract_mesh()
            del gsExtractor_end
            torch.cuda.empty_cache()

            save_path_start = os.path.join(mesh_path, f"start_{mask_id}.ply")
            save_path_end = os.path.join(mesh_path, f"end_{mask_id}.ply")
            o3d.io.write_triangle_mesh(save_path_start, mesh_start)
            o3d.io.write_triangle_mesh(save_path_end, mesh_end)
            
            del mesh_start, mesh_end
            torch.cuda.empty_cache()

        need_gpu = mask_id == -1 and (eval_app or visualize)
        if need_gpu:
            rgbs = torch.stack(rgbs_start + rgbs_end, 0).cuda()
            depths = torch.stack(depths_start + depths_end, 0).cuda() if visualize else None
            gt_rgbs = torch.stack(gt_rgbs_start + gt_rgbs_end, 0)
        else:
            rgbs = torch.stack(rgbs_start + rgbs_end, 0)
            depths = torch.stack(depths_start + depths_end, 0)
            gt_rgbs = torch.stack(gt_rgbs_start + gt_rgbs_end, 0)
        
        if visualize and not need_gpu:
            depths = depths.cuda()

        if mask_id == -1 and eval_app:
            PSNR, SSIM, LPIPS = 0, 0, 0
            for rgb, gt_rgb in zip(rgbs, gt_rgbs):
                PSNR += psnr(rgb[None], gt_rgb[None])
                SSIM += ssim_func(rgb[None], gt_rgb[None])
                LPIPS += lpips(rgb[None], gt_rgb[None])
            n = len(rgbs)
            PSNR, SSIM, LPIPS = PSNR / n, SSIM / n, LPIPS / n
            PSNR, SSIM, LPIPS = PSNR.item(), SSIM.item(), LPIPS.item()
            if not visualize:
                del rgbs, gt_rgbs
                if depths is not None:
                    del depths
                torch.cuda.empty_cache()
    
    gt_path = args.source_path + "/gt"

    pred_joint_list = [{}] + pred_joint_list
    mesh_files = [f"meshes/start_{i}.ply" for i in range(len(pred_joint_list))]
    export_joint_info_json(pred_joint_list, mesh_files, save_dir)
    
    output = pd.DataFrame()
    if name == 'train':
        print("Evaluating CD")
        s, d_list, w = eval_CD_2states(gt_path, mesh_path, [list(range(1, 1+num_d_joints))[idx] for idx in real_perm], n_trials=3)
        mean_s = (s['start'] + s['end']) / 2
        mean_d_list = [(ds + de) / 2 for ds, de in zip(d_list['start'], d_list['end'])]
        mean_w = (w['start'] + w['end']) / 2
        # save result as csv
        output['avg_CD_static'] = [mean_s]
        output['avg_CD_whole'] = [mean_w]
        output['avg_CD_dynamic'] = 0
        output['avg_angle'] = 0
        output['avg_distance'] = 0
        output['avg_theta_diff'] = 0
        for i, d in enumerate(mean_d_list):
            output[f'CD_dynamic_{i}'] = [d]
            output[f'angle_{i}'] = [results_list[i][0]]
            output[f'distance_{i}'] = [results_list[i][1]]
            output[f'theta_diff_{i}'] = [results_list[i][2]]
            output['avg_CD_dynamic'] += d
            output['avg_angle'] += results_list[i][0]
            output['avg_distance'] += results_list[i][1]
            output['avg_theta_diff'] += results_list[i][2]
        output['avg_CD_dynamic'] /= len(mean_d_list)
        output['avg_angle'] /= len(mean_d_list)
        output['avg_distance'] /= len(mean_d_list)
        output['avg_theta_diff'] /= len(mean_d_list)
        for state in ['start', 'end']:
            output[f'{state}_CD_static'] = [s[state]]
            output[f'{state}_CD_whole'] = [w[state]]
            for i, d in enumerate(d_list[state]):
                output[f'{state}_CD_dynamic_{i}'] = [d]
    output['PSNR'] = [PSNR]
    output['SSIM'] = [SSIM]
    output['LPIPS'] = [LPIPS]
    output = output.transpose()
    print(output)
    output.to_csv(os.path.join(model_path, name, "ours_{}".format(iteration), "result.csv"), index=True)


def render_sets(args, dataset: ModelParams, iteration, pipeline: PipelineParams, skip_train: bool, skip_test: bool, mode: str, load2device_on_the_fly=False):
    with torch.no_grad():
        deform = DeformModel(dataset)
        loaded = deform.load_weights(dataset.model_path, iteration=iteration)
        if not loaded:
            raise ValueError(f"Failed to load weights from {dataset.model_path}")
        num_joints = len(deform.deform.joint_types)
        gaussians = GaussianModel(dataset.sh_degree, num_joints=num_joints)
        dataset.model_path_noseed = os.path.join('outputs', args.dataset, args.subset, args.scene_name)
        scene = Scene(dataset, gaussians, load_iteration=iteration, is_test=True)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        cam_traj = scene.getTrainCameras()
        if mode == 'render':
            cam_traj = generate(cam_traj)

        if not skip_train:
            render_set(args, load2device_on_the_fly, "train", scene.loaded_iter, cam_traj, gaussians, pipeline, background, deform, eval_app=args.eval_app, visualize=False)

        if not skip_test:
            render_set(args, load2device_on_the_fly, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, deform, eval_app=args.eval_app, visualize=False)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Visualize the rendered images")
    parser.add_argument("--eval_app", action="store_true", help="Evaluate the rendered images with PSNR, SSIM, and LPIPS")
    parser.add_argument("--mode", default='eval', choices=['render', 'eval'])
    parser.add_argument("--seed", default=0, type=int)
    args = get_combined_args(parser)
    args.source_path = f'data/{args.dataset}/{args.subset}/{args.scene_name}'
    _lm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arguments", "larger_motion_state.json")
    with open(_lm_path, "r", encoding="utf-8") as f:
        _larger_map = json.load(f)
    args.larger_motion_state = int(_larger_map[args.dataset][args.subset][args.scene_name])
    print("Rendering " + args.source_path + ' with '+ args.model_path)
    safe_state(args.quiet)
    seed_everything(args.seed)
    render_sets(args, model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode, load2device_on_the_fly=args.load2gpu_on_the_fly)
