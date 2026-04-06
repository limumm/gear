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

from scene.cameras import Camera
import numpy as np
import torch
import cv2
from utils.general_utils import PILtoTorch, ArrayToTorch
from utils.graphics_utils import fov2focal
import json

WARNED = False


def loadCam(args, id, cam_info, resolution_scale, flow_dirs):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    # if len(cam_info.image.split()) > 3:
    #     # import torch
    #     # resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
    #     loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
    # else:
    #     # resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    #     loaded_mask = None

    resized_mask_np = cv2.resize(cam_info.mask, resolution, interpolation=cv2.INTER_NEAREST) if cam_info.mask is not None else None
    resized_mask = torch.from_numpy(resized_mask_np) if resized_mask_np is not None else torch.tensor([], dtype=torch.float32).to(args.data_device)

    depth = cam_info.depth
    if depth is not None:
        depth = cv2.resize(depth, resolution, interpolation=cv2.INTER_NEAREST)
    mono_depth = cam_info.mono_depth
    if mono_depth is not None:
        mono_depth = cv2.resize(mono_depth, resolution, interpolation=cv2.INTER_NEAREST)

    gt_image = resized_image_rgb[:3, ...]

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    cam = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, mask=resized_mask, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id,
                  data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu', fid=cam_info.fid,
                  depth=depth, mono_depth=mono_depth, flow_dirs=flow_dirs)
    return cam


def load_correspondence(args, cam, corr_dir):
    if corr_dir is None:
        return
    H = cam.image_height * args.resolution if args.resolution != -1 else cam.image_height
    corr_list = np.load(corr_dir, allow_pickle=True)['data']
    def rev_pixel(pixel, H):
        return pixel
        # return pixel * np.array([1, -1]).reshape(1, 2) + np.array([0, H - 1]).reshape(1, 2)

    for corr in corr_list:
        src_name, tgt_name = list(corr.keys())
        src_pixel, tgt_pixel = corr[src_name], corr[tgt_name]  # smaller coords are at the top - the same index to use for images
        src_pixel = torch.from_numpy(rev_pixel(src_pixel, H)).long()
        tgt_pixel = torch.from_numpy(rev_pixel(tgt_pixel, H)).long()
        tgt_id = tgt_name.split('_')[1]
        cam.corr[tgt_id] = [src_pixel, tgt_pixel]


def cameraList_from_camInfos(cam_infos, resolution_scale, args, flow_dirs_list=None):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, None if flow_dirs_list is None else flow_dirs_list[id]))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def camera_nerfies_from_JSON(path, scale):
    """Loads a JSON camera into memory."""
    with open(path, 'r') as fp:
        camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
        camera_json['tangential_distortion'] = camera_json['tangential']

    return dict(
        orientation=np.array(camera_json['orientation']),
        position=np.array(camera_json['position']),
        focal_length=camera_json['focal_length'] * scale,
        principal_point=np.array(camera_json['principal_point']) * scale,
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.array(camera_json['radial_distortion']),
        tangential_distortion=np.array(camera_json['tangential_distortion']),
        image_size=np.array((int(round(camera_json['image_size'][0] * scale)),
                             int(round(camera_json['image_size'][1] * scale)))),
    )
