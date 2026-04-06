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
import random
import json

import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos
from utils.other_utils import  get_larger_motion_state


class Scene:
    gaussians: GaussianModel
    def __init__(self, args: ModelParams, gaussians: GaussianModel, gaussians1: GaussianModel=None, load_iteration=None, init_with_random_pcd=False, resolution_scales=[1.0], load_path=None, is_train=False, is_test=False, coarse_name='coarse_point_cloud'):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.model_path_noseed = args.model_path_noseed 
        self.loaded_iter = None
        self.gaussians = gaussians
        self.gaussians1 = gaussians1 if gaussians1 is not None else None
        self.args = args
        self.coarse_name = coarse_name
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        if os.path.exists(os.path.join(args.source_path, "transforms_train_start.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            raise ValueError("No scene info file found!")
    
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Cameras extent: ", self.cameras_extent)
        print("Loading Cameras")
        if not is_train:
            self.train_cameras = [{},{}] # 0: start, 1: end
            self.orig_cameras = [{},{}] # support state 0 and 1
            self.test_cameras = [{},{}] # 0: start, 1: end
            for i in range(2):
                for resolution_scale in resolution_scales:
                    self.train_cameras[i][resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras_2s[i], resolution_scale, args)
                    self.test_cameras[i][resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras_2s[i], resolution_scale, args)

            if load_path and os.path.exists(load_path):
                print(f"Loading PLY from specified path: {load_path}")
                self.gaussians.load_ply(load_path)
            elif self.loaded_iter and not is_test:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                    "point_cloud_0.ply"))
                self.gaussians1.load_ply(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                    f"point_cloud_1.ply"))
            elif is_test:
                self.gaussians.load_ply(os.path.join(self.model_path, 
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                    "point_cloud.ply"))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
                self.gaussians1.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        else:
            self.larger_motion_state = int(load_path.split("_")[-1].split(".")[0])
            sup_state = 0 if self.larger_motion_state == 1 else 1
            self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras_2s[sup_state], 1.0, args)
            self.orig_cameras = cameraList_from_camInfos(scene_info.train_cameras_2s[self.larger_motion_state], 1.0, args)
            self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras_2s[sup_state], 1.0, args)

            self.gaussians.load_ply(os.path.join(self.model_path_noseed,
                                                    self.coarse_name,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                    f"point_cloud_{self.larger_motion_state}.ply"))
            self.gaussians1.load_ply(os.path.join(self.model_path_noseed,
                                                    self.coarse_name,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                    f"point_cloud_{sup_state}.ply"), with_grad=False)

    def save(self, iteration, is_best=False):
        if is_best:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_best")
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            with open(os.path.join(point_cloud_path, "iter.txt"), 'w') as f:
                f.write(f"iteration: {iteration}")
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_2gs(self, iteration, num_slots, vis_cano=True, vis_center=True):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_0.ply"))
        self.gaussians1.save_ply(os.path.join(point_cloud_path, "point_cloud_1.ply"))
        larger_motion_state = get_larger_motion_state(os.path.join(point_cloud_path, "point_cloud.ply"), num_slots, vis_cano, voxel_size=0.02)
        with open(os.path.join(point_cloud_path, "larger_motion_state.txt"), "w") as f:
            f.write(str(larger_motion_state))
    
    def getTrainCameras_start(self, scale=1.0):
        return self.train_cameras[0][scale]
    
    def getTrainCameras_end(self, scale=1.0):
        return self.train_cameras[1][scale]
    
    def getTestCameras_start(self, scale=1.0):
        return self.test_cameras[0][scale]
    
    def getTestCameras_end(self, scale=1.0):
        return self.test_cameras[1][scale]

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[0][scale] + self.train_cameras[1][scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[0][scale] + self.test_cameras[1][scale]
