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

import os
from PIL import Image
from typing import NamedTuple, Optional
from utils.graphics_utils import getWorld2View2
import numpy as np
import json
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None
    mono_depth: Optional[np.array] = None
    mask: Optional[np.array] = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    train_cameras_2s: list
    test_cameras_2s: list


def getNerfppNorm(cam_info, apply=False):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    cam_centers = []
    if apply:
        c2ws = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        if apply:
            c2ws.append(C2W)
        cam_centers.append(C2W[:3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal
    translate = -center
    if apply:
        c2ws = np.stack(c2ws, axis=0)
        c2ws[:, :3, -1] += translate
        c2ws[:, :3, -1] /= radius
        w2cs = np.linalg.inv(c2ws)
        for i in range(len(cam_info)):
            cam = cam_info[i]
            cam_info[i] = cam._replace(R=w2cs[i, :3, :3].T, T=w2cs[i, :3, 3])
        apply_translate = translate
        apply_radius = radius
        translate = 0
        radius = 1.
        return {"translate": translate, "radius": radius, "apply_translate": apply_translate, "apply_radius": apply_radius}
    else:
        return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                        vertices['blue']]).T / 255.0
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.zeros((positions.shape[0], 3))
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", no_bg=False, load_depth=True, load_mono_depth=True, load_mask=True):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        fovy = contents["camera_angle_y"]

        frames = contents["frames"]
        # frames = sorted(frames, key=lambda x: int(os.path.basename(x['file_path']).split('.')[0].split('_')[-1]))
        frames = sorted(frames, key=lambda x: x['file_path'])
        for idx, frame in enumerate(frames):
            cam_name = frame["file_path"]
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.join(path, cam_name))), 'rgba')):
                cam_name = os.path.join(os.path.dirname(os.path.dirname(os.path.join(path, cam_name))), 'rgba', os.path.basename(cam_name)).replace('.jpg', '.png')
            if cam_name.endswith('jpg') or cam_name.endswith('png'):
                cam_name = os.path.join(path, cam_name)
            else:
                cam_name = os.path.join(path, cam_name + extension)
            frame_time = frame['time']

            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            try:
                im_data = np.array(image.convert("RGBA"))
            except:
                print(f'{image_path} is damaged')
                continue

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] 
            if no_bg:
                norm_data[:, :, :3] = norm_data[:, :, 3:4] * norm_data[:, :, :3] + bg * (1 - norm_data[:, :, 3:4])
            
            arr = np.concatenate([arr, mask], axis=-1)

            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGBA" if arr.shape[-1] == 4 else "RGB")

            FovY = fovy
            FovX = fovx

            idx = str(int(image_name)).zfill(3)
            depth_path = image_path.replace('rgba', 'depth')
            if load_depth and os.path.exists(depth_path):
                depth = cv.imread(depth_path, -1) / 1e3
                h, w = depth.shape
                if depth.size == mask.size:
                    depth[mask[..., 0] < 0.5] = 0
                else:
                    depth[cv.resize(mask[..., 0], [w, h], interpolation=cv.INTER_NEAREST) < 0.5] = 0
                depth[depth < 0.1] = 0
            else:
                depth = None

            mono_depth_path = image_path.replace('rgba', 'mono_depth')
            if load_mono_depth and os.path.exists(mono_depth_path):
                mono_depth = cv.imread(mono_depth_path, cv.IMREAD_GRAYSCALE) / 255
                h, w = mono_depth.shape
                if mono_depth.size == mask.size:
                    mono_depth[mask[..., 0] < 0.5] = 0
                else:
                    mono_depth[cv.resize(mask[..., 0], [w, h], interpolation=cv.INTER_NEAREST) < 0.5] = 0
            else:
                mono_depth = None


            mask_path = image_path.replace('rgba', 'mask')
            mask_path = mask_path.replace('.png', '.npy')
            if load_mask and os.path.exists(mask_path):
                mask_sam = np.load(mask_path)
                # Apply alpha mask filtering to SAM mask if needed
                # Zero out regions where alpha < 0.5 to ensure foreground-only masks
                alpha_mask = mask[..., 0]  # in [0, 1]
                mh, mw = (mask_sam.shape[:2]) if mask_sam.ndim >= 2 else (alpha_mask.shape[0], alpha_mask.shape[1])
                if alpha_mask.shape != (mh, mw):
                    alpha_resized = cv.resize(alpha_mask, (mw, mh), interpolation=cv.INTER_NEAREST)
                else:
                    alpha_resized = alpha_mask
                # Preserve dtype of loaded SAM mask
                mask_sam = np.where(alpha_resized >= 0.5, mask_sam, 0).astype(mask_sam.dtype)
            else:
                mask_sam = None

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,mono_depth=mono_depth,mask=mask_sam,
                                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], fid=frame_time))

    return cam_infos


def readInfo_2states(path, white_background, eval, extension=".png", no_bg=True):
    print("Reading Training Transforms")
    train_cam_infos = []
    test_cam_infos = []
    for state in ['start', 'end']:
        train_infos = readCamerasFromTransforms(
            path, f"transforms_train_{state}.json", white_background, extension, no_bg=no_bg)
        try:
            test_infos = readCamerasFromTransforms(
                path, f"transforms_test_{state}.json", white_background, extension, no_bg=no_bg)
        except:
            test_infos = []
        if not eval:
            train_infos.extend(test_infos)
        train_cam_infos.append(train_infos)
        test_cam_infos.append(test_infos)
        print(f"Read train_{state} transforms with {len(train_infos)} cameras")
        print(f"Read  test_{state} transforms with {len(test_infos)} cameras")

    nerf_normalization = getNerfppNorm(train_cam_infos[0] + train_cam_infos[1])

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos[0] + train_cam_infos[1],
                           test_cameras=test_cam_infos[0] + test_cam_infos[1],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           train_cameras_2s=train_cam_infos,
                           test_cameras_2s=test_cam_infos)
    return scene_info


sceneLoadTypeCallbacks = {
    "Blender": readInfo_2states,
}
