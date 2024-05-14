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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            # 用于根据缩放和旋转构建协方差矩阵。
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, dataset):
        """
        类中定义了一个名为 setup_functions 的方法，用于设置模型的激活函数。
            在该方法中，首先定义了一个名为 build_covariance_from_scaling_rotation 的内部函数，用于根据缩放和旋转构建协方差矩阵。
        接着，将指定的激活函数赋值给类的属性，包括缩放的激活函数 scaling_activation、缩放的反激活函数 scaling_inverse_activation、
            协方差矩阵的激活函数 covariance_activation、不透明度的激活函数 opacity_activation 和旋转的激活函数 rotation_activation。
        在类的构造函数 __init__ 中，首先初始化类的属性，包括 active_sh_degree、max_sh_degree、_xyz、_features_dc、_features_rest、_scaling、_rotation、
            _opacity、max_radii2D、xyz_gradient_accum、denom、optimizer、percent_dense 和 spatial_lr_scale。
        最后，调用 setup_functions 方法，设置模型的激活函数。
        """
        self.dataset = dataset  # 存放相关参数用于控制不同的if分支
        self.active_sh_degree = 0
        self.max_sh_degree = dataset.sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

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
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

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
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """
        这段代码定义了一个名为 `create_from_pcd` 的方法，用于从点云数据创建相机。
        方法接受两个参数：`pcd` 表示点云数据，`spatial_lr_scale` 表示空间低分辨率缩放比例。
        首先，将传入的点云数据 `pcd` 的点坐标转换为 `torch.tensor` 类型，并将结果转移到 cuda 设备上，并将结果赋值给 `fused_point_cloud`。
        然后，将点云数据 `pcd` 的颜色值转换为球谐系数表示的颜色值，并将结果转移到 cuda 设备上，并将结果赋值给 `fused_color`。
        接着，创建一个大小为 `(fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)` 的零张量 `features`，并将其转移到 cuda 设备上。
        然后，将 `fused_color` 的前三个通道的值赋值给 `features` 的第一个通道的第一个元素，将 `features` 的第二个通道的其他元素赋值为 0.0。
        然后，打印初始化时的点的数量。
        接下来，计算点云数据 `pcd` 中点之间的距离的平方，并将结果转移到 cuda 设备上，并将结果赋值给 `dist2`。
        然后，通过 `clamp_min` 方法将 `dist2` 中的值夹在一个最小值为 0.0000001 的范围内。
        接着，计算点云数据 `pcd` 中点之间的距离的对数，并将结果转移到 cuda 设备上，并将结果赋值给 `scales`。
        然后，创建一个大小为 `(fused_point_cloud.shape[0], 4)` 的零张量 `rots`，并将其转移到 cuda 设备上。
        然后，将 `rots` 的第一列的值设置为 1，表示四元数的实部。
        接着，创建一个大小为 `(fused_point_cloud.shape[0], 1)` 的零张量 `opacities`，并将其转移到 cuda 设备上。
        然后，将 `opacities` 的值设置为经过反 sigmoid 函数处理的 0.1。
        接下来，将 `fused_point_cloud`、`features`、`scales`、`rots` 和 `opacities` 分别转换为可训练的参数，
            并分别赋值给 `self._xyz`、`self._features_dc`、`self._features_rest`、`self._scaling`、`self._rotation` 和 `self._opacity`。
        最后，创建一个大小为 `(self.get_xyz.shape[0])` 的零张量 `max_radii2D`，并将其转移到 cuda 设备上。
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()  # [Point_num, 3] 点云3D坐标
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())  # [Point_num, 3] 将点云数据 `pcd` 的颜色值转换为球谐系数表示的颜色值
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()  # [Point_num, 3, 16] 特征有16个分量，第一个分量存储初始sh系数
        features[:, :3, 0] = fused_color  # 第三个维度的第一个size全为fused_color，其他维度全赋值为0
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])  # 初始化的点云数量
        # TODO: debug simple_knn 这个库，distCUDA2输入参数的size=[Point_num, 3], 即每个3D点的坐标
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)  # [Point_num,] torch.clamp_min设置一个下限min，tensor中有元素小于这个值, 就把对应的值赋为min
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)  # [Point_num, 3]
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")  # [Point_num, 4]
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))  # [Point_num, 1]

        # TODO：记录初始点云所处的坐标范围，在后面进行致密化时限定clone和split的点云要在一定范围内，不能随意增加，从而减小不必要的点云，需要进一步调试
        if self.dataset.limited_range > 0:
            print("Limiting range to ", self.dataset.limted_range)
            min_x, max_x = torch.min(fused_point_cloud[:, 0]), torch.max(fused_point_cloud[:, 0])  # 获取初始点云的范围
            min_y, max_y = torch.min(fused_point_cloud[:, 1]), torch.max(fused_point_cloud[:, 1])
            min_z, max_z = torch.min(fused_point_cloud[:, 2]), torch.max(fused_point_cloud[:, 2])
            # 考虑到点云不一定非要局限在初始范围内，可以适当的扩展初始的范围
            scale = self.dataset.limited_range  # 范围扩展的倍数
            self.limited_range = [min_x * scale, max_x * scale,
                                  min_y * scale, max_y * scale,
                                  min_z * scale, max_z * scale]

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))  # 记住点云坐标需要计算梯度
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))  # [Point_num, 1, 3] 记录特征的第一个分量点球谐系数，需要计算梯度
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))  # [Point_num, 15, 3] 记录特征的剩余15个分量的值，需要计算梯度
        self._scaling = nn.Parameter(scales.requires_grad_(True))  # [Point_num, 3] 记录每个点云在x y z三个方向的尺度因子，需要计算梯度
        self._rotation = nn.Parameter(rots.requires_grad_(True))  # [Point_num, 4] 记录每个点云的旋转矩阵，需要计算梯度
        self._opacity = nn.Parameter(opacities.requires_grad_(True))  # [Point_num, 1] 记录每个点云的透明度，需要计算梯度
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # [Point_num, 1]
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # [Point_num, 1]

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]  # 单独为每个参数配置相应的学习率

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)  # 得到学习率更新策略

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step
         每步学习率调度'''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

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
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

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

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        这个函数的目的是将一个字典中的张量按照优化器的参数组进行分类，并将可优化张量附加到优化器中，同时更新优化器的状态。以下是对函数的详细解释：
        首先，函数开始时创建一个空的字典 optimizable_tensors，用于存储可优化张量。
        遍历优化器的参数组（self.optimizer.param_groups），对于每个参数组：
            a. 从字典 tensors_dict 中获取与当前参数组关联的张量（extension_tensor）。
            b. 检查优化器的状态中是否已经存在与当前参数组关联的exp_avg和exp_avg_sq张量。
            c. 如果存在，将extension_tensor添加到现有的exp_avg和exp_avg_sq张量中，并更新它们的状态。
            d. 从优化器的状态中删除与当前参数组关联的param_groups。
            e. 将拼接后的张量（包括原有的和附加的）转换为nn.Parameter对象，并将其添加到优化器的param_groups中。
            f. 将当前参数组的名称和新的nn.Parameter对象添加到optimizable_tensors字典中。
        循环结束后，返回optimizable_tensors字典。
        :param tensors_dict:
        :return:
        """
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
                              new_rotation):
        """
        该函数的目的是对输入的新的三维坐标（new_xyz）、特征（new_features_dc）、不透明度（new_opacities）、
        缩放（new_scaling）和旋转（new_rotation）进行密度优化。
        实现过程中，它将输入的张量合并成一个新的张量，然后将这个新的张量传递给优化器进行优化。
        :param new_xyz: 确保输入的张量（new_xyz、new_features_dc、new_features_rest、new_opacities、new_scaling和new_rotation）在传递给函数之前已经进行了必要的预处理，例如归一化、缩放等。
        :param new_features_dc:
        :param new_features_rest:
        :param new_opacities:
        :param new_scaling:
        :param new_rotation:
        :return:
        """
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition 提取满足梯度条件的点
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)  # [23509,] 比场景范围大到一定程度的点采用split，选取出大于阈值的点云

        if self.dataset.limited_range > 0:  # 只有在限制点云的split区域后才进行区域筛选
            selected_pts_mask_limited_range = self.densify_and_limited_range(self.get_xyz)  # 对需要split的点进行筛选，确保其范围在场景范围内
            selected_pts_mask = torch.logical_and(selected_pts_mask, selected_pts_mask_limited_range)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)  # [24, 3]选出这个几个高斯点云对应的标准差，同时复制N份，也就是2份
        means = torch.zeros((stds.size(0), 3), device="cuda")  # [24, 3] 生成均值，均值为0
        samples = torch.normal(mean=means, std=stds)  # [24, 3] 使用正态分布复制点云
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)  # [24, 3, 3] 得到筛选出的高斯点云的旋转矩阵3x3
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)  # [24, 3] 先按照split的mask对点云进行分割，然后再限定点云的范围
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))  # 将筛选出来的点的索引和新的点云中的点的索引拼接在一起，用于后续的剪枝操作，因为复制出来的点不用再split，因此用False填充
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition 提取满足梯度条件的点
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)  # 比场景范围小到一定程度的点采用clone

        if self.dataset.limited_range > 0:  # 只有在限制点云的clone区域后才进行区域筛选
            selected_pts_mask_limited_range = self.densify_and_limited_range(self._xyz)  # 对需要克隆的点进行筛选，确保其范围在场景范围内
            selected_pts_mask = torch.logical_and(selected_pts_mask, selected_pts_mask_limited_range)  # 将前后两次的筛选结果进行与操作

        new_xyz = self._xyz[selected_pts_mask]  # 将筛选出来的点云的坐标复制到新变量，实现对点的克隆
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)

    def densify_and_limited_range(self, xyz):
        selected_pts_mask_min_x = torch.where(xyz[:, 0] >= self.limited_range[0], True, False)
        selected_pts_mask_max_x = torch.where(xyz[:, 0] <= self.limited_range[1], True, False)
        selected_pts_mask_x = torch.logical_and(selected_pts_mask_min_x, selected_pts_mask_max_x)

        selected_pts_mask_min_y = torch.where(xyz[:, 1] >= self.limited_range[2], True, False)
        selected_pts_mask_max_y = torch.where(xyz[:, 1] <= self.limited_range[3], True, False)
        selected_pts_mask_y = torch.logical_and(selected_pts_mask_min_x, selected_pts_mask_max_x)

        selected_pts_mask_min_z = torch.where(xyz[:, 2] >= self.limited_range[4], True, False)
        selected_pts_mask_max_z = torch.where(xyz[:, 2] <= self.limited_range[5], True, False)
        selected_pts_mask_z = torch.logical_and(selected_pts_mask_min_x, selected_pts_mask_max_x)

        selected_pts_mask = torch.logical_and(torch.logical_and(selected_pts_mask_x, selected_pts_mask_y), selected_pts_mask_z)

        return selected_pts_mask
        # self._xyz = self._xyz[selected_pts_mask]  # 将筛选出来的点云的坐标复制到新变量
        # self._features_dc = self._features_dc[selected_pts_mask]
        # self._features_rest = self._features_rest[selected_pts_mask]
        # self._opacity = self._opacity[selected_pts_mask]
        # self._scaling = self._scaling[selected_pts_mask]
        # self._rotation = self._rotation[selected_pts_mask]
        #
        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
        #                            new_rotation)


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        # self.densify_and_limited_range()  # 限制clone和split生成的点云范围，该步骤会迅速增加显存用量，需要查找原因：可能是由于优化器多次cat变量的原因导致后期点云数量迅速增大

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        该函数的目的是在训练神经网络时，对观察空间点的张量（viewspace_point_tensor）进行密度优化统计。
        实现原理是将每个更新过滤器的xyz梯度累加到self.xyz_gradient_accum中，同时将每个更新过滤器的denom值累加1。
        :param viewspace_point_tensor:
        :param update_filter:
        :return:
        """
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)  # 将点云中各个点的x y z三个方向的梯度求2范数
        self.denom[update_filter] += 1


if __name__ == '__main__':
    pass