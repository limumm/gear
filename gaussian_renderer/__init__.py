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

import torch
import math
from scene.gaussian_model import GaussianModel
from utils.point_utils import depth_to_normal
from utils.dual_quaternion import quaternion_mul
import seaborn as sns
import numpy as np
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz= None, d_rot= None, scaling_modifier=1.0, random_bg_color=False, scale_const=None, mask=None, vis_mask=None, train_coarse=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # d_xyz= None
    # d_rot= None
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    bg = bg_color if not random_bg_color else torch.rand_like(bg_color)
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    if train_coarse:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            nb_mask=0,
            # pipe.debug
        )
    else:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            nb_mask=pc.num_joints,
            debug=False,
            # pipe.debug
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # Ensure compatibility when nb_mask == 0: provide empty CUDA tensor for mask logits
    nb_mask_value = getattr(rasterizer.raster_settings, 'nb_mask', 0)
    if nb_mask_value == 0:
        rasterizer.mask_logits = torch.Tensor([]).cuda()
    else:
        rasterizer.mask_logits = pc.mask_logits[vis_mask] if vis_mask is not None else pc.mask_logits
    xyz = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    sh_features = pc.get_features

    means3D = xyz + d_xyz if d_xyz is not None else xyz
    means2D = screenspace_points 
    if scale_const is not None:
        opacity = torch.ones_like(pc.get_opacity)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(d_rot, scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = quaternion_mul(d_rot, rotations) if d_rot is not None else rotations
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    if mask != None:
        shs = None
        pallete = torch.from_numpy(np.array(sns.color_palette("hls", mask.max() + 1))).float().to(pc.get_xyz.device)
        colors_precomp = pallete[mask]
    else:
        shs = pc.get_features
        colors_precomp = None
    if scale_const is not None:
        scales = scale_const * torch.ones_like(scales)
    # Rasterize visible Gaussians to image.
    if vis_mask is not None:
        means3D = means3D[vis_mask]
        means2D = means2D[vis_mask]
        shs = shs[vis_mask] if shs is not None else None
        colors_precomp = colors_precomp[vis_mask] if colors_precomp is not None else None
        opacity = opacity[vis_mask]
        scales = scales[vis_mask]
        rotations = rotations[vis_mask]
        cov3D_precomp = cov3D_precomp[vis_mask] if cov3D_precomp is not None else None
    
    raster_out = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    if isinstance(raster_out, tuple) and len(raster_out) == 4:
        rendered_image, radii, allmap, mask_logits = raster_out
    else:
        rendered_image, radii, allmap = raster_out
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "mask_logits": mask_logits if isinstance(raster_out, tuple) and len(raster_out) == 4 else None,
    }


    # additional regularizations
    render_alpha = allmap[1:2]
    # label_map = torch.max(mask_logits[:7],dim=0).values*40

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()


    rets.update({
            'alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'depth': surf_depth,
            'surf_normal': surf_normal,
    })

    return rets