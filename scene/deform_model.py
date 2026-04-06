import os

import torch

from scene.artgs import ArtGS
from utils.general_utils import get_expon_lr_func
from utils.system_utils import searchForMaxIteration


class DeformModel:
    def __init__(self, args, points_num=0):
        self.deform = ArtGS(args, points_num).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    @property
    def reg_loss(self):
        return self.deform.reg_loss

    def step(self, gaussians, is_training=True):
        return self.deform(gaussians, is_training=is_training)

    def train_setting(self, training_args):
        base_lr = (
            training_args.position_lr_init
            * self.spatial_lr_scale
            * training_args.deform_lr_scale
        )
        final_lr = training_args.position_lr_final * training_args.deform_lr_scale
        l = [
            {
                "params": group["params"],
                "lr": base_lr,
                "name": group["name"],
            }
            for group in self.deform.trainable_parameters()
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=base_lr,
            lr_final=final_lr,
            lr_delay_mult=training_args.position_lr_delay_mult,
            begin_steps=20000,
            max_steps=training_args.deform_lr_max_steps,
        )

    def save_weights(self, model_path, iteration, is_best=False):
        subdir = "iteration_best" if is_best else f"iteration_{iteration}"
        out_weights_path = os.path.join(model_path, "deform", subdir)
        os.makedirs(out_weights_path, exist_ok=True)
        save_dict = {
            "state_dict": self.deform.state_dict(),
            "joint_types": self.deform.joint_types,
        }
        torch.save(save_dict, os.path.join(out_weights_path, "deform.pth"))
        if is_best:
            with open(os.path.join(out_weights_path, "iter.txt"), "w") as f:
                f.write(f"iteration: {iteration}")

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration

        weights_path = os.path.join(
            model_path, "deform", f"iteration_{loaded_iter}", "deform.pth"
        )
        if os.path.exists(weights_path):
            return self._load_checkpoint(weights_path)

        best_weights_path = os.path.join(model_path, "deform", "iteration_best", "deform.pth")
        if os.path.exists(best_weights_path):
            print(f"iteration_{loaded_iter} not found, loading best weights instead")
            return self._load_checkpoint(best_weights_path)

        return False

    def _load_checkpoint(self, weights_path):
        checkpoint = torch.load(weights_path)
        self.deform.load_state_dict(checkpoint["state_dict"])
        if "joint_types" in checkpoint:
            self.deform.joint_types = checkpoint["joint_types"]
            print(f"Loaded joint_types: {self.deform.joint_types}")
        return True

    def update_learning_rate(self, iteration):
        deform_names = ("deform", "mlp", "qr_s", "qd_s", "joints")
        for param_group in self.optimizer.param_groups:
            name = param_group["name"]
            if name in deform_names or name == "slot":
                param_group["lr"] = self.deform_scheduler_args(iteration)

    def set_joint_parameters_trainable(self, trainable=True):
        self.deform.set_joint_parameters_trainable(trainable)

    def get_optimization_state(self):
        return self.deform.get_current_optimization_state()

    def get_optimization_summary(self):
        return self.deform.get_optimization_summary()

    def print_optimization_status(self):
        print("=== DeformModel Optimization Status ===")
        print(self.get_optimization_summary())
        print("=====================================")

    def reinitialize_optimizer(self, training_args):
        """Rebuild Adam after joint-type changes; copies Adam state for same-named param groups."""
        if self.optimizer is None:
            print("Warning: No existing optimizer to reinitialize")
            return

        old_state = self.optimizer.state_dict()
        base_lr = (
            training_args.position_lr_init
            * self.spatial_lr_scale
            * training_args.deform_lr_scale
        )
        l = [
            {
                "params": group["params"],
                "lr": base_lr,
                "name": group["name"],
            }
            for group in self.deform.trainable_parameters()
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        new_state = self.optimizer.state_dict()
        for param_group in new_state["param_groups"]:
            param_name = param_group["name"]
            for old_param_group in old_state["param_groups"]:
                if old_param_group["name"] != param_name:
                    continue
                old_param_id = old_param_group["params"][0]
                new_param_id = param_group["params"][0]
                if old_param_id in old_state["state"]:
                    new_state["state"][new_param_id] = old_state["state"][old_param_id]
                break

        self.optimizer.load_state_dict(new_state)
        print(f"Reinitialized optimizer with {len(l)} parameter groups")

    def switch_joint_types(self, joint_info_list, training_args=None):
        """
        Args:
            joint_info_list: New joint type list (e.g. after r->p switch).
            training_args: If set and optimizer exists, rebuild optimizer after switch.
        """
        success = self.deform.switch_revolute_to_prismatic(joint_info_list, training_args)
        if success and training_args is not None and self.optimizer is not None:
            print("Reinitializing optimizer after joint type switch...")
            self.reinitialize_optimizer(training_args)
        return success
