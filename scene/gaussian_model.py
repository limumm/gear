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

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn

from simple_knn._C import distCUDA2
from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    build_scaling_rotation_inverse,
    get_expon_lr_func,
    inverse_sigmoid,
)
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH, SH2RGB
from utils.system_utils import mkdir_p


class GaussianModel:

    def setup_functions(self, dynamic_threshold_ratio):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.dynamic_threshold_ratio = dynamic_threshold_ratio

    def __init__(self, sh_degree: int, fea_dim=0, with_motion_mask=False, dynamic_threshold_ratio=0.02, num_joints=0):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        
        self.denom = torch.empty(0)
        
        self.num_joints = num_joints
        self.fea_dim = num_joints if num_joints > 0 else 20
        self.mask_logits = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions(dynamic_threshold_ratio)
    def param_names(self):
        return ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity', 'max_radii2D', 'xyz_gradient_accum']

    @classmethod
    def build_from(cls, gs, **kwargs):
        new_gs = GaussianModel(**kwargs)
        new_gs._xyz = nn.Parameter(gs._xyz)
        new_gs._features_dc = nn.Parameter(torch.zeros_like(gs._features_dc))
        new_gs._features_rest = nn.Parameter(torch.zeros_like(gs._features_rest))
        new_gs._scaling = nn.Parameter(gs._scaling)
        new_gs._rotation = nn.Parameter(gs._rotation)
        new_gs._opacity = nn.Parameter(gs._opacity)
        new_gs.feature = nn.Parameter(gs.feature)
        new_gs.max_radii2D = torch.zeros((new_gs.get_xyz.shape[0]), device="cuda")
        return new_gs
    
    @property
    def joint_probs(self):
        """Softmax over joint (mask) logits, shape [N, fea_dim]."""
        return torch.softmax(self.mask_logits, dim=-1)
    
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    def copy(self):
        new_gs = GaussianModel(self.max_sh_degree, self.fea_dim, self.dynamic_threshold_ratio, self.num_joints)
        new_gs._xyz = self._xyz.clone()
        new_gs._features_dc = self._features_dc.clone()
        new_gs._features_rest = self._features_rest.clone()
        new_gs._scaling = self._scaling.clone()
        new_gs._rotation = self._rotation.clone()
        new_gs._opacity = self._opacity.clone()
        new_gs.max_radii2D = self.max_radii2D.clone()
        new_gs.xyz_gradient_accum = self.xyz_gradient_accum.clone()
        new_gs.denom = self.denom.clone()
        new_gs.mask_logits = self.mask_logits.clone()
        new_gs.spatial_lr_scale = self.spatial_lr_scale
        new_gs.num_joints = self.num_joints
        return new_gs
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.num_joints) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_rgb(self):
        return SH2RGB(self._features_dc.squeeze(1))

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, d_rot, scaling_modifier=1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def get_covariance_inv(self):
        L = build_scaling_rotation_inverse(self.get_scaling, self._rotation)
        actual_covariance_inv = L @ L.transpose(1, 2)
        return actual_covariance_inv

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float=5., print_info=True, max_point_num=150_000):
        self.spatial_lr_scale = spatial_lr_scale
        if type(pcd.points) == np.ndarray:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        else:
            fused_point_cloud = pcd.points
        if type(pcd.colors) == np.ndarray:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        else:
            fused_color = pcd.colors
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        if print_info:
            print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.mask_logits = torch.zeros((self._xyz.shape[0], self.fea_dim), dtype=torch.float32, device="cuda", requires_grad=True)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.spatial_lr_scale = 5

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self.mask_logits], 'lr': training_args.feature_lr, 'name': 'mask_logits'}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        """Step LR for the xyz parameter group."""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def cancel_grad(self):
        self._xyz.requires_grad = False
        self._opacity.requires_grad = False
        self._features_dc.requires_grad = False
        self._features_rest.requires_grad = False
        self._scaling.requires_grad = False
        self._rotation.requires_grad = False

    def regain_grad(self):
        self._xyz = nn.Parameter(self._xyz.data, requires_grad=True)
        self._opacity = nn.Parameter(self._opacity.data, requires_grad=True)
        self._features_dc = nn.Parameter(self._features_dc.data, requires_grad=True)
        self._features_rest = nn.Parameter(self._features_rest.data, requires_grad=True)
        self._scaling = nn.Parameter(self._scaling.data, requires_grad=True)
        self._rotation = nn.Parameter(self._rotation.data, requires_grad=True)

        if hasattr(self, "optimizer"):
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "opacity":
                    param_group['params'] = [self._opacity]
                elif param_group["name"] == "f_dc":
                    param_group['params'] = [self._features_dc]
                elif param_group["name"] == "f_rest":
                    param_group['params'] = [self._features_rest]
                elif param_group["name"] == "scaling":
                    param_group['params'] = [self._scaling]
                elif param_group["name"] == "rotation":
                    param_group['params'] = [self._rotation]
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self.fea_dim):
            l.append('fea_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()

        rotation_quat = self._rotation.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        try:
            from scipy.spatial.transform import Rotation as R

            rotation_matrices = R.from_quat(rotation_quat).as_matrix()
            normals = rotation_matrices[:, 2, :]
        except ImportError:
            print("Warning: scipy not found; computing normals from quaternions manually")
            for i, quat in enumerate(rotation_quat):
                w, x, y, z = quat
                rotation_matrix = np.array([
                    [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                    [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                    [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
                ])
                normals[i] = rotation_matrix[:, 2]
        
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        if self.fea_dim > 0:
            feature = self.mask_logits.detach().cpu().numpy()
            attributes = np.concatenate((attributes, feature), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        mask_logits = self.mask_logits.detach()
        joint_ids = mask_logits.argmax(-1)
        unique_joint_ids = joint_ids.unique()
        for j_id in unique_joint_ids:
            if j_id == 0:
                continue
            inds = joint_ids == j_id
            count = inds.sum().item()
            if count < 2000:
                opacities_new[inds] = self.get_opacity[inds]

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, with_grad=True):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        self.normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                        np.asarray(plydata.elements[0]["ny"]),
                        np.asarray(plydata.elements[0]["nz"])), axis=1)
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        fea_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("fea")]
        if len(fea_names) > 0:
            feas = np.zeros((xyz.shape[0], len(fea_names)))
            for idx, attr_name in enumerate(fea_names):
                feas[:, idx] = np.asarray(plydata.elements[0][attr_name])

            if len(fea_names) != self.fea_dim:
                print(f"Warning: Feature dimension mismatch. PLY has {len(fea_names)}, model expects {self.fea_dim}")
                if len(fea_names) < self.fea_dim:
                    feas_padded = np.zeros((xyz.shape[0], self.fea_dim))
                    feas_padded[:, : len(fea_names)] = feas
                    feas = feas_padded
                else:
                    feas = feas[:, : self.fea_dim]
        else:
            feas = np.zeros((xyz.shape[0], self.fea_dim))

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(with_grad))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                with_grad))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                with_grad))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(with_grad))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(with_grad))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(with_grad))
        
        self.mask_logits = nn.Parameter(torch.tensor(feas, dtype=torch.float, device="cuda").requires_grad_(with_grad))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def load_ply_cano(self, path, state=0.5):
        # state = 0
        if abs(state - 0.5) < 1e-6:
            self.load_ply(path)
            print("Loaded canonical state = 0.5")
        elif abs(state) < 1e-6:
            cano_path = path.replace('point_cloud.ply', 'point_cloud_0.ply')
            self.load_ply(cano_path)
            print("Loaded canonical state = 0")
        elif abs(state - 1) < 1e-6:
            cano_path = path.replace('point_cloud.ply', 'point_cloud_1.ply')
            self.load_ply(cano_path)
            print("Loaded canonical state = 1")
        else:
            raise ValueError("Invalid canonical state")
        xyz_static_0 = torch.tensor(np.load(path.replace('point_cloud.ply', 'xyz_static_0.npy'))).float().to(self._xyz.device)
        xyz_static_1 = torch.tensor(np.load(path.replace('point_cloud.ply', 'xyz_static_1.npy'))).float().to(self._xyz.device)
        xyz_dynamic_0 = torch.tensor(np.load(path.replace('point_cloud.ply', 'xyz_dynamic_0.npy'))).float().to(self._xyz.device)
        xyz_dynamic_1 = torch.tensor(np.load(path.replace('point_cloud.ply', 'xyz_dynamic_1.npy'))).float().to(self._xyz.device)
        xyzs = [torch.cat([xyz_static_0, xyz_dynamic_0]), torch.cat([xyz_static_1, xyz_dynamic_1])]
        return xyzs

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.mask_logits = optimizable_tensors["mask_logits"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_feature=None):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}
        
        d["mask_logits"] = new_feature

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.mask_logits = optimizable_tensors["mask_logits"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads=None, grad_threshold=None, scene_extent=None, N=2, selected_pts_mask=None, without_prune=False):
        if selected_pts_mask is None:
            n_init_points = self.get_xyz.shape[0]
            # Extract points that satisfy the gradient condition
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grads.shape[0]] = grads.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling,
                                                            dim=1).values > self.percent_dense * scene_extent)
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_feature = self.mask_logits[selected_pts_mask].repeat(N, 1) if self.fea_dim > 0 else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_feature)

        if not without_prune:
            prune_filter = torch.cat(
                (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
            self.prune_points(prune_filter)

    def densify_and_clone(self, grads=None, grad_threshold=None, scene_extent=None, selected_pts_mask=None):
        # Extract points that satisfy the gradient condition
        if selected_pts_mask is None:
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling,
                                                            dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_feature = self.mask_logits[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_feature)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask_opacity = (self.get_opacity < min_opacity).squeeze()
        prune_mask = prune_mask_opacity
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask_opacity, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1