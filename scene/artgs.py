import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scene.gaussian_model import GaussianModel
from utils.dual_quaternion import *
from scene.module import gumbel_softmax, ProgressiveBandHashGrid
from utils.dual_quaternion import matrix_to_quaternion
import os

class ArtGS(nn.Module):
    def __init__(self, args, points_num=0):
        super().__init__()
        self.points_num = points_num
        self.slot_size = args.slot_size
        self.joint_types = args.joint_types.split(",")
        self.num_slots = len(self.joint_types)

        self.center_from_coarse_pose = torch.zeros(self.num_slots, 3, device="cuda", dtype=torch.float32)
        self.init_joints_from_coarse_pose(args)
                
        self.register_buffer('qr_s', torch.Tensor([1, 0, 0, 0]))
        self.register_buffer('qd_s', torch.Tensor([0, 0, 0, 0]))

            
    def slotdq_to_gsdq(self, slot_qr, slot_qd, mask):
        # slot_qr: [K, 4], slot_qd: [K, 4], mask: [N, K]
        qr = torch.einsum('nk, kl->nl', mask, slot_qr)   # [N, 4]
        qd = torch.einsum('nk, kl->nl', mask, slot_qd)   # [N, 4]
        return normalize_dualquaternion(qr, qd)
    
    def get_slot_deform(self):
        qrs = []
        qds = []
        for i, joint_type in enumerate(self.joint_types):
            if i == 0:
                assert joint_type == 's'
                qr, qd = self.qr_s, self.qd_s
            else:
                joint = self.joints[i]
                qr = F.normalize(joint[:4], p=2, dim=-1) # normalize quaternion
                t0 = torch.cat([torch.zeros(1).to(qr.device), joint[4:7]]) # translation
                # handle joint type
                if joint_type == 'p':  # prismatic joint
                    # force rotation quaternion to be identity quaternion, not participate in gradient calculation
                    qr = torch.tensor([1., 0., 0., 0.], device=qr.device, requires_grad=False)
                    qd = 0.5 * quaternion_mul(t0, qr)
                else:  # revolute joint
                    qd = 0.5 * quaternion_mul(t0, qr)
            qrs.append(qr)
            qds.append(qd)
        qrs, qds = torch.stack(qrs), torch.stack(qds)
        return qrs, qds

    def deform_pts(self, xc, mask, slot_qr, slot_qd, current_phase):
        """deform the points, without using state"""
        mask = mask[:, :self.num_slots]
        if current_phase == 'M':
            # set the position with the maximum probability to 1, the rest to 0
            max_indices = torch.argmax(mask, dim=1, keepdim=True) # [N, 1]
            mask = torch.zeros_like(mask).scatter_(1, max_indices, 1.0) # [N, K+1]
        gs_qr, gs_qd = self.slotdq_to_gsdq(slot_qr, slot_qd, mask)
        xt = dual_quaternion_apply((gs_qr, gs_qd), xc)
        return xt, gs_qr
    
    def trainable_parameters(self):
        params = []
        # add joints parameters to trainable parameters
        if hasattr(self, 'joints') and self.joints is not None:
            params.append({'params': self.joints, 'name': 'joints'})
        return params
    
    def get_mask_from_logits(self, gaussians: GaussianModel, is_training=True):
        # """get mask from logits of gaussians"""
        if gaussians.num_joints > 0:
            # use softmax to get probability distribution
            probs = gaussians.joint_probs  # [N, K+1]
            return probs
        else:
            # if there is no logit vector, return default mask
            N = gaussians.get_xyz.shape[0]
            mask = torch.zeros(N, self.num_slots, device=gaussians.get_xyz.device)
            mask[:, 0] = 1.0  # default all points belong to static part
            return mask
            
    def deform_pts_with_interpolation(self, xc, mask, slot_qr, slot_qd):
        N = mask.shape[0]
        K_plus_1 = self.num_slots

        transformed_positions = torch.zeros(N, K_plus_1, 3, device=xc.device)
        transformed_rotations = torch.zeros(N, K_plus_1, 4, device=xc.device)

        for k in range(K_plus_1):
            current_qr = slot_qr[k:k+1] # [1, 4]
            current_qd = slot_qd[k:k+1] # [1, 4]

            if k == 0:
                transformed_positions[:, k, :] = xc
                transformed_rotations[:, k, :] = torch.tensor([1., 0., 0., 0.], device=xc.device).expand(N, -1)
                continue

            if k <= self.num_slots and self.joint_types[k] == 'p':
                interp_qr = torch.tensor([1., 0., 0., 0.], device=xc.device).unsqueeze(0)  # [1, 4]
                interp_qd = current_qd
            else:
                interp_qr = current_qr
                interp_qd = current_qd

            gs_qr_k = interp_qr.expand(N, -1)
            gs_qd_k = interp_qd.expand(N, -1)

            transformed_positions[:, k, :] = dual_quaternion_apply((gs_qr_k, gs_qd_k), xc)
            transformed_rotations[:, k, :] = gs_qr_k

        weights = mask[:, :self.num_slots].unsqueeze(-1) # [N, K+1, 1]

        xt = torch.sum(weights * transformed_positions, dim=1) # [N, 3]

        rot = torch.sum(weights * transformed_rotations, dim=1) # [N, 4]
        rot = F.normalize(rot, p=2, dim=-1)

        return xt, rot, transformed_positions, transformed_rotations

    @torch.no_grad()
    def get_joint_param(self):
        qrs, qds = self.get_slot_deform()
        qrs, qds = qrs[1:], qds[1:]
        joint_info_list = []
        for i, joint_type in enumerate(self.joint_types[1:]):
            qr, qd = qrs[i], qds[i]
            qr, t = dual_quaternion_to_quaternion_translation((qr, qd))
            R = quaternion_to_matrix(qr).cpu().numpy()
            t = t.cpu().numpy()
            
            if joint_type == 'r': # revolute joint
                axis_dir, theta = quaternion_to_axis_angle(qr)
                axis_dir, theta = axis_dir.cpu().numpy(), theta.cpu().numpy()
                try:
                    # try to compute axis position
                    I_minus_R = np.eye(3) - R
                    axis_position = np.matmul(np.linalg.inv(I_minus_R), t.reshape(3, 1)).reshape(-1)
                    axis_position += axis_dir * np.dot(axis_dir, -axis_position)
                except np.linalg.LinAlgError:
                    # # when matrix is not invertible, use default position
                    # print(f"Warning: Failed to compute axis position for joint {i}, using default position")
                    axis_position = np.zeros(3)
                
                # R = R @ R
                t = R @ t + t
                joint_info = {'type': joint_type,
                            'axis_position': axis_position,
                            'axis_direction': axis_dir,
                            'theta': np.rad2deg(theta),
                            'rotation': R, 'translation': t}
            elif joint_type == 'p':
                # t = t * 2
                theta = np.linalg.norm(t)
                if theta < 1e-8:
                    # handle case when translation is near zero or very small
                    print(f"Warning: Joint {i} has near-zero translation, using default direction")
                    axis_dir = np.array([0., 0., 1.])
                else:
                    axis_dir = t / theta
                joint_info = {'type': joint_type,
                            'axis_position': np.zeros(3), 
                            'axis_direction': axis_dir, 
                            'theta': theta,
                            'rotation': R, 'translation': t}
            joint_info_list.append(joint_info)
        return joint_info_list
    
    def switch_revolute_to_prismatic(self, joint_info_list, training_args=None):
        if len(joint_info_list) != len(self.joint_types):
            print(f"Error: joint_info_list length {len(joint_info_list)} != joint_types length {len(self.joint_types)}")
            return False
        
        # check if there are joints to switch
        changed_joints = []
        for i, (old_type, new_type) in enumerate(zip(self.joint_types, joint_info_list)):
            if old_type == 'r' and new_type == 'p':
                changed_joints.append(i)
        
        if not changed_joints:
            print("No revolute joints to switch to prismatic")
            return True
        
        print(f"Switching joints {changed_joints} from revolute to prismatic")
        
        # update joint types
        self.joint_types = joint_info_list.copy()
        
        # for joints to switch, modify their parameters
        if hasattr(self, 'joints') and self.joints is not None:
            with torch.no_grad():
                for joint_idx in changed_joints:
                    if joint_idx > 0 and joint_idx - 1 < self.joints.shape[0]:  # skip static part (joint_idx=0)
                        # set rotation quaternion to identity quaternion [1, 0, 0, 0]
                        self.joints[joint_idx, :4] = torch.tensor([1.0, 0.0, 0.0, 0.0], 
                                                                        device=self.joints.device, 
                                                                        dtype=self.joints.dtype)
                        # keep translation part unchanged, let optimizer continue optimizing
                        print(f"  Joint {joint_idx}: Set rotation quaternion to identity, keeping translation")
        
        print(f"Successfully switched joints to: {self.joint_types}")
        return True

    def one_transform(self, gaussians:GaussianModel, joint_types, is_training, current_phase):
        xc = gaussians.get_xyz#.detach()
        N = xc.shape[0]
        
        # get mask
        mask = self.get_mask_from_logits(gaussians)  # [N, K+1]
        
        qr, qd = self.get_slot_deform()  # get joint parameters, dual quaternion
        
        xt, rot = self.deform_pts(xc, mask, qr, qd, current_phase)

        d_xyz = xt - xc
        d_rotation = rot.detach()

        return {
            'd_xyz': d_xyz,
            'd_rotation': d_rotation,
            'xt': xt,
            'mask': mask.argmax(-1),
        }
    
    def forward(self, gaussians: GaussianModel, is_training=False):
        xc = gaussians._xyz.detach()
        N = xc.shape[0]
        
        # get mask
        mask = self.get_mask_from_logits(gaussians)  # [N, K+1]

        qr, qd = self.get_slot_deform()

        # deform the points, without using state
        xt, rot, _, _ = self.deform_pts_with_interpolation(xc, mask, qr, qd)
        
        d_xyz = xt - xc
        d_rotation = rot.detach()
        d_values = {
            'd_xyz': d_xyz,
            'd_rotation': d_rotation,
            'xt': xt,
            'mask': mask.argmax(-1),
            # 'per_joint_xt': None,
            # 'per_joint_rot': None,
        }

        return [d_values]
    
    def interpolate(self, gaussians: GaussianModel, time_list):
        xc = gaussians._xyz.detach()
        mask = self.get_mask_from_logits(gaussians) # [N, K+1]
        qr1, qd1 = self.get_slot_deform()
        # initialize as identity quaternion
        qr0, qd0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=qr1.device, dtype=qr1.dtype), torch.tensor([0.0, 0.0, 0.0, 0.0], device=qd1.device, dtype=qd1.dtype)

        dx_list = []
        dr_list = []
        for t in time_list:
            slot_qr = (1 - t) * qr0 + t * qr1
            slot_qd = (1 - t) * qd0 + t * qd1
            gs_qr, gs_qd = self.slotdq_to_gsdq(slot_qr, slot_qd, mask)
            xt = dual_quaternion_apply((gs_qr, gs_qd), xc)
            dx_list.append(xt - xc)
            dr_list.append(gs_qr)
        return dx_list, dr_list

    def _default_joint_tensor(self):
        """(num_slots, 7): row 0 unused in get_slot_deform (static); rows 1.. hold dual-quaternion params. Matches train.init_logits_from_voxelized_joints."""
        joints = torch.zeros(self.num_slots, 7, device="cuda", dtype=torch.float32)
        joints[:, 0] = 1.0
        if self.num_slots > 1:
            joints[1:, :] = joints[1:, :] + torch.randn(self.num_slots - 1, 7, device="cuda", dtype=torch.float32) * 1e-5
        return joints

    def init_joints_from_coarse_pose(self, args):
        """
        Load joint_poses.npy (or args.joint_poses) when present; else default tensor.
        Training overwrites from joint_voxel_info.npy when available.
        """
        joints = self._default_joint_tensor()
        coarse_pose_path = os.path.join(
            args.model_path, "point_cloud", f"iteration_{args.iterations}", "joint_poses.npy"
        )
        if getattr(args, "joint_poses", None) is not None:
            joint_poses = args.joint_poses
        elif os.path.isfile(coarse_pose_path):
            joint_poses = np.load(coarse_pose_path, allow_pickle=True)
        else:
            self.joints = nn.Parameter(joints)
            return

        if joint_poses is None or len(joint_poses) == 0:
            self.joints = nn.Parameter(joints)
            return

        dev = joints.device
        dtype = joints.dtype
        try:
            for i, joint_pose in enumerate(joint_poses):
                row = i + 1
                if row >= self.num_slots:
                    print(f"Warning: skipping extra coarse joint {i} (num_slots={self.num_slots}).")
                    break
                center = torch.tensor(
                    joint_pose["source_cluster_center"], device=dev, dtype=dtype
                )
                self.center_from_coarse_pose[row] = center

                if joint_pose["type"] == "r":
                    R = torch.tensor(joint_pose["R"], device=dev, dtype=dtype)
                    t = torch.tensor(joint_pose["t"], device=dev, dtype=dtype)
                    joints[row, :4] = matrix_to_quaternion(R)
                    joints[row, 4:7] = t
                elif joint_pose["type"] == "p":
                    t = torch.tensor(joint_pose["translation"], device=dev, dtype=dtype)
                    joints[row, :4] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev, dtype=dtype)
                    joints[row, 4:7] = t
                else:
                    print(f"Warning: unknown coarse joint type '{joint_pose['type']}', row {row} left at default.")

            print(f"Initialized {min(len(joint_poses), self.num_slots - 1)} dynamic joints from coarse joint_poses.")
        except Exception as e:
            print(f"Warning: failed to parse coarse joint_poses ({e}); default joint init.")
            joints = self._default_joint_tensor()

        self.joints = nn.Parameter(joints)

    def get_current_optimization_state(self):
        """get current optimization state information"""
        state_info = {
            'joints_trainable': False,
            'qr_s_trainable': False,
            'qd_s_trainable': False,
            'mask_trainable': False
        }
        
        if hasattr(self, 'joints'):
            state_info['joints_trainable'] = self.joints.requires_grad
        if hasattr(self, 'qr_s'):
            state_info['qr_s_trainable'] = self.qr_s.requires_grad
        if hasattr(self, 'qd_s'):
            state_info['qd_s_trainable'] = self.qd_s.requires_grad
        
        return state_info
    
    def set_joint_parameters_trainable(self, trainable=True):
        """set joints parameters trainable"""
        if hasattr(self, 'joints'):
            self.joints.requires_grad_(trainable)
        if hasattr(self, 'qr_s'):
            self.qr_s.requires_grad_(trainable)
        if hasattr(self, 'qd_s'):
            self.qd_s.requires_grad_(trainable)
        
        status = "trainable" if trainable else "frozen"
        # print(f"Joint parameters set to {status}")
    
    def get_optimization_summary(self):
        """get optimization summary information"""
        summary = []
        summary.append(f"Joint types: {self.joint_types}")
        summary.append(f"Total joints: {self.num_slots} (1 static + {self.num_slots-1} dynamic)")
        
        if hasattr(self, 'joints'):
            summary.append(f"Joint parameters: {self.joints.shape}")
            summary.append(f"Joint parameters trainable: {self.joints.requires_grad}")
        
        if hasattr(self, 'qr_s'):
            summary.append(f"qr_s trainable: {self.qr_s.requires_grad}")
        if hasattr(self, 'qd_s'):
            summary.append(f"qd_s trainable: {self.qd_s.requires_grad}")
        
        return "\n".join(summary)