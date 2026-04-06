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

os.environ["MODEL_TYPE"] = "coarse"

import torch
import torch.nn.functional as F
import torchvision
import tqdm
from argparse import ArgumentParser
from pytorch_lightning import seed_everything

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.dataset_readers import fetchPly
from utils.general_utils import get_linear_noise_func, safe_state
from utils.loss_utils import l1_loss, ssim
from utils.log_utils import prepare_output_and_logger
from utils.metrics import *


class Trainer:
    def __init__(self, args, dataset, opt, pipe):
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.dataset.model_path_noseed = args.model_path
        self.tb_writer = prepare_output_and_logger(dataset)

        self.gaussians = [
            GaussianModel(dataset.sh_degree, fea_dim=0, with_motion_mask=True),
            GaussianModel(dataset.sh_degree, fea_dim=0, with_motion_mask=True),
        ]

        self.scene = Scene(
            dataset,
            self.gaussians[0],
            self.gaussians[1],
            load_iteration=None,
            init_with_random_pcd=True,
        )

        if args.init_from_pcd and self.scene.loaded_iter is None:
            print("Init Gaussians with pcd from depth.")
            for i, state in enumerate(["start", "end"]):
                self.gaussians[i].create_from_pcd(
                    fetchPly(f"{args.source_path}/point_cloud_{state}.ply")
                )

        for i in (0, 1):
            self.gaussians[i].training_setup(opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iteration = 1 if self.scene.loaded_iter is None else self.scene.loaded_iter

        self.ema_loss_for_log = 0.0
        self.ema_dist_for_log = 0.0
        self.ema_normal_for_log = 0.0

        self.viewpoint_stacks = [
            self.scene.getTrainCameras_start(),
            self.scene.getTrainCameras_end(),
        ]
        self.progress_bar = tqdm.tqdm(
            range(self.iteration - 1, opt.iterations), desc="Training progress"
        )
        self.smooth_term = get_linear_noise_func(
            lr_init=0.1,
            lr_final=1e-15,
            lr_delay_mult=0.01,
            max_steps=20000,
        )

        self.reg_weight = self.args.opacity_reg_weight
        self.metric_depth_loss_weight = args.metric_depth_loss_weight

    def train(self, iters=5000):
        for _ in tqdm.trange(iters, desc="Training"):
            self.train_step()

    def train_step(self):
        # --- Prefetch views and GT to GPU (non_blocking) ---
        if self.iteration % self.opt.oneupSHdegree_step == 0:
            self.gaussians[0].oneupSHdegree()
            self.gaussians[1].oneupSHdegree()

        idx0 = randint(0, len(self.viewpoint_stacks[0]) - 1)
        view_0 = self.viewpoint_stacks[0][idx0]
        idx1 = randint(0, len(self.viewpoint_stacks[1]) - 1)
        view_1 = self.viewpoint_stacks[1][idx1]

        random_bg_0 = (
            not self.dataset.white_background
            and self.opt.random_bg_color
            and view_0.gt_alpha_mask is not None
        )
        bg_0 = self.background if not random_bg_0 else torch.rand_like(self.background).cuda()

        random_bg_1 = (
            not self.dataset.white_background
            and self.opt.random_bg_color
            and view_1.gt_alpha_mask is not None
        )
        bg_1 = self.background if not random_bg_1 else torch.rand_like(self.background).cuda()

        gt_image_0 = view_0.original_image.cuda(non_blocking=True)
        gt_alpha_0 = view_0.gt_alpha_mask.cuda(non_blocking=True)
        gt_image_1 = view_1.original_image.cuda(non_blocking=True)
        gt_alpha_1 = view_1.gt_alpha_mask.cuda(non_blocking=True)

        # --- Back-to-back renders for both articulation states ---
        d_xyz, d_rot = None, None
        pkg_0 = render(
            view_0, self.gaussians[0], self.pipe, bg_0,
            d_xyz=d_xyz, d_rot=d_rot, train_coarse=True,
        )
        pkg_1 = render(
            view_1, self.gaussians[1], self.pipe, bg_1,
            d_xyz=d_xyz, d_rot=d_rot, train_coarse=True,
        )

        # --- Loss, backward, densify per state ---
        total_loss_log = 0.0
        total_dist_log = 0.0
        total_normal_log = 0.0

        states_data = [
            (0, pkg_0, view_0, gt_image_0, gt_alpha_0, bg_0, random_bg_0),
            (1, pkg_1, view_1, gt_image_1, gt_alpha_1, bg_1, random_bg_1),
        ]

        lambda_normal = self.opt.lambda_normal if self.iteration > 7000 else 0.0
        lambda_dist = self.opt.lambda_dist if self.iteration > 3000 else 0.0

        for state_idx, pkg, view, gt_img, gt_alpha, bg, rnd_bg in states_data:
            image = pkg["render"]

            if rnd_bg:
                gt_img = gt_alpha * gt_img + (1 - gt_alpha) * bg[:, None, None]
            elif self.dataset.white_background and gt_alpha is not None:
                gt_img = gt_alpha * gt_img + (1 - gt_alpha) * self.background[:, None, None]

            Ll1 = l1_loss(image, gt_img)
            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (
                1.0 - ssim(image, gt_img)
            )

            dist_loss = lambda_dist * pkg["rend_dist"].mean()
            normal_error = (1 - (pkg["rend_normal"] * pkg["surf_normal"]).sum(dim=0))[None]
            normal_loss = lambda_normal * normal_error.mean()
            loss = loss + dist_loss + normal_loss

            if view.depth is not None and self.metric_depth_loss_weight > 0:
                depth = pkg["depth"]
                gt_depth = view.depth.cuda(non_blocking=True)
                invalid_mask = (gt_depth < 0.1) & (gt_alpha > 0.5)
                valid_mask = ~invalid_mask
                n_valid_pixel = valid_mask.sum()
                if n_valid_pixel > 100:
                    depth_loss = (
                        torch.log(1 + torch.abs(depth - gt_depth)) * valid_mask
                    ).sum() / n_valid_pixel
                    loss = loss + depth_loss * self.metric_depth_loss_weight

            opacity = self.gaussians[state_idx].get_opacity
            reg_loss = F.binary_cross_entropy(opacity, (opacity.detach() > 0.5).float())
            loss = loss + reg_loss * self.reg_weight

            loss.backward()

            total_loss_log += loss.item()
            total_dist_log += dist_loss.item()
            total_normal_log += normal_loss.item()

            with torch.no_grad():
                current_gs = self.gaussians[state_idx]

                if current_gs.max_radii2D.shape[0] == 0:
                    current_gs.max_radii2D = torch.zeros_like(pkg["radii"])

                visibility_filter = pkg["visibility_filter"]
                radii = pkg["radii"]
                current_gs.max_radii2D[visibility_filter] = torch.max(
                    current_gs.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

                if self.iteration < self.opt.densify_until_iter:
                    current_gs.add_densification_stats(pkg["viewspace_points"], visibility_filter)

                    if (
                        self.iteration > self.opt.densify_from_iter
                        and self.iteration % self.opt.densification_interval == 0
                    ):
                        size_threshold = (
                            20 if self.iteration > self.opt.opacity_reset_interval else None
                        )
                        current_gs.densify_and_prune(
                            self.opt.densify_grad_threshold,
                            0.005,
                            self.scene.cameras_extent,
                            size_threshold,
                        )

                    if self.iteration % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background
                        and self.iteration == self.opt.densify_from_iter
                    ):
                        current_gs.reset_opacity()

                if self.iteration < self.opt.iterations:
                    current_gs.optimizer.step()
                    current_gs.update_learning_rate(self.iteration)
                    current_gs.optimizer.zero_grad(set_to_none=True)

        # --- Logging (EMA over two states) ---
        with torch.no_grad():
            avg_loss = total_loss_log / 2.0
            avg_dist = total_dist_log / 2.0
            avg_normal = total_normal_log / 2.0

            self.ema_loss_for_log = 0.4 * avg_loss + 0.6 * self.ema_loss_for_log
            self.ema_dist_for_log = 0.4 * avg_dist + 0.6 * self.ema_dist_for_log
            self.ema_normal_for_log = 0.4 * avg_normal + 0.6 * self.ema_normal_for_log

            if self.iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{self.ema_loss_for_log:.{5}f}",
                    "dist": f"{self.ema_dist_for_log:.{4}f}",
                    "norm": f"{self.ema_normal_for_log:.{4}f}",
                    "Pts": f"{len(self.gaussians[0].get_xyz)}",
                }
                self.progress_bar.set_postfix(loss_dict)
                self.progress_bar.update(10)

            if self.iteration == self.opt.iterations:
                self.progress_bar.close()
                self.scene.save_2gs(
                    self.iteration,
                    self.args.num_slots,
                    self.args.vis_cano,
                    self.args.vis_center,
                )

        self.iteration += 1


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--vis_cano", action="store_true", default=True)
    parser.add_argument("--vis_center", action="store_true", default=True)
<<<<<<< HEAD
    parser.add_argument("--seed", type=int, default=0)
=======
    parser.add_argument('--seed', type=int, default=0)
>>>>>>> main

    args = parser.parse_args(sys.argv[1:])
    args.source_path = f"{args.source_path}/{args.dataset}/{args.subset}/{args.scene_name}"

    try:
        with open("./arguments/num_slots.json", "r", encoding="utf-8") as f:
            args.num_slots = json.load(f)[args.dataset][args.subset][args.scene_name]
    except Exception as e:
        print(f"Warning: Could not load num_slots from json: {e}, defaulting to 1 or checking args.")

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    seed_everything(args.seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    trainer = Trainer(
        args=args,
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
    )
    trainer.train(args.iterations)
    print("\nTraining complete.")
