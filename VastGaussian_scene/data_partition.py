# Author: Peilun Kang
# Contact: kangpeilun@nefu.edu.cn
# License: Apache Licence
# Project: VastGaussian
# File: data_partition.py
# Time: 5/15/24 2:28 PM
# Des: 数据划分策略
"""
因为不理解如何进行曼哈顿世界对齐，使世界坐标的y轴垂直于地平面，因此本实现假设已经是对其的坐标
"""

import os
import numpy as np
from typing import NamedTuple

from scene.dataset_readers import CameraInfo, storePly
from utils.graphics_utils import getWorld2View2, BasicPointCloud


class CameraPose(NamedTuple):
    camera: CameraInfo
    pose: np.array  # [x, y, z] 坐标


class CameraPartition(NamedTuple):
    partition_id: str  # 部分的编号
    cameras: list      # 该部分对应的相机
    point_cloud: BasicPointCloud  # 该部分对应的点云
    ori_bbox: list     # 该部分原始的边界坐标 [x_min, x_max, z_min, z_max]
    extend_bbox: list  # 按照extend_rate拓展后边界的坐标


class ProgressiveDataPartitioning:
    # 渐进数据分区
    def __init__(self, scene_info, model_path, m_region=2, n_region=4, extend_rate=0.2):
        self.scene_info = scene_info
        self.model_path = model_path   # 存放模型位置
        self.partition_dir = os.path.join(model_path, "partition_point_cloud")
        self.partition_ori_dir = os.path.join(self.partition_dir, "ori")
        self.partition_extend_dir = os.path.join(self.partition_dir, "extend")
        self.m_region = m_region
        self.n_region = n_region

        if not os.path.exists(self.partition_ori_dir): os.makedirs(self.partition_ori_dir)        # 创建存放分块后 拓展前 点云的文件夹
        if not os.path.exists(self.partition_extend_dir): os.makedirs(self.partition_extend_dir)  # 创建存放分块后 拓展后 点云的文件夹

    def Camera_position_based_region_division(self):
        """1.基于相机位置的区域划分
        思路：1.首先将整个场景的相机坐标投影到以xz轴组成的平面上
             2.按照x轴方向，将所有的相机分成m部分
             3.按照z轴方向，将每一部分分成n部分 (默认将整个区域分成2*4=8个部分),同时保证mxn个部分中的相机数量的均衡
             4.返回每个部分的边界坐标，以及每个部分对应的相机
        """
        m, n = self.m_region, self.n_region
        train_cameras = self.scene_info.train_cameras
        CameraPose_list = []
        for idx, camera in enumerate(train_cameras):
            W2C = getWorld2View2(camera.R, camera.T)  # 根据相机的旋转矩阵和平移向量，将 世界坐标系 -> 相机坐标系
            C2W = np.linalg.inv(W2C)  # 相机坐标系 -> 世界坐标系
            CameraPose_list.append(CameraPose(camera=camera, pose=C2W[:3, 3:4]))  # 将变换矩阵中的平移向量作为相机的中心，在世界坐标系下相机的中心坐标

        # 2.沿着x轴将相机分成m部分
        m_partition_dict = {}
        total_camera = len(CameraPose_list)  # 获取相机总数
        num_of_camera_per_m_partition = total_camera // m  # m个部分，每部分相机数量
        sorted_CameraPose_by_x_list = sorted(CameraPose_list, key=lambda x: x.pose[0])  # 按照x轴坐标排序
        for i in range(m):  # 按照x轴将所有相机分成m部分
            m_partition_dict[str(i+1)] = sorted_CameraPose_by_x_list[i*num_of_camera_per_m_partition:(i+1)*num_of_camera_per_m_partition]
        if total_camera % m != 0:  # 如果相机数量不是m的整数倍，则将余下的相机直接添加到最后一部分
            m_partition_dict[str(m)].extend(sorted_CameraPose_by_x_list[m*num_of_camera_per_m_partition:])

        # 3.沿着z轴将相机分成n部分
        partition_dict = {}  # 保存mxn每个部分的相机数量
        for partition_idx, camera_list in m_partition_dict.items():
            partition_total_camera = len(camera_list)  # m个部分，每部分相机数量
            num_of_camera_per_n_partition = partition_total_camera // n  # n个部分，每部分相机数量
            sorted_CameraPose_by_z_list = sorted(camera_list, key=lambda x: x.pose[2])  # 按照z轴坐标排序
            for i in range(n):  # 按照z轴将所有相机分成n部分
                partition_dict[f"{partition_idx}_{i+1}"] = sorted_CameraPose_by_z_list[i*num_of_camera_per_n_partition:(i+1)*num_of_camera_per_n_partition]
            if partition_total_camera % n != 0:  # 如果相机数量不是n的整数倍，则将余下的相机直接添加到最后一部分
                partition_dict[f"{partition_idx}_{n}"].extend(sorted_CameraPose_by_z_list[n*num_of_camera_per_n_partition:])

        self.partition_dict = partition_dict


    def extract_point_cloud(self, pcd, bbox):
        partition_mask = (pcd.points[:, 0] >= bbox[0]) & (pcd.points[:, 0] <= bbox[1]) & (
                    pcd.points[:, 2] >= bbox[2]) & (pcd.points[:, 2] <= bbox[3])  # 筛选在范围内的点云，得到对应的mask
        points = pcd.points[partition_mask]
        colors = pcd.colors[partition_mask]
        normals = pcd.normals[partition_mask]
        return points, colors, normals

    def Position_based_data_selection(self, extend_rate=0.2):
        """2.基于位置的数据选择
        思路：1.计算每个partition的x z边界
             2.然后按照extend_rate将每个partition的边界坐标扩展，得到新的边界坐标 [x_min, x_max, z_min, z_max]
             3.根据extend后的边界坐标，获取该部分对应的点云
        """
        # 计算每个部分的拓展后的边界坐标，以及该部分对应的点云
        pcd = self.scene_info.point_cloud
        partition_list = []
        point_num = 0
        point_extend_num = 0
        for partition_idx, camera_list in self.partition_dict.items():
            min_x, max_x = min(camera.pose[0] for camera in camera_list)[0], max(camera.pose[0] for camera in camera_list)[0]
            min_z, max_z = min(camera.pose[2] for camera in camera_list)[0], max(camera.pose[2] for camera in camera_list)[0]
            ori_bbox = [min_x, max_x, min_z, max_z]
            extend_bbox = [min_x-extend_rate*(max_x-min_x), max_x+extend_rate*(max_x-min_x), min_z-extend_rate*(max_z-min_z), max_z+extend_rate*(max_z-min_z)]
            # extend_bbox = ori_bbox
            # 获取该部分对应的点云
            points, colors, normals = self.extract_point_cloud(pcd, ori_bbox)  # 分别获取原始边界的点云，和拓展边界后的点云
            points_extend, colors_extend, normals_extend = self.extract_point_cloud(pcd, extend_bbox)

            partition_list.append(CameraPartition(partition_id=partition_idx, cameras=camera_list, ori_bbox=ori_bbox, extend_bbox=extend_bbox,
                                                  point_cloud=BasicPointCloud(points_extend, colors_extend, normals_extend)))
            point_num += points.shape[0]
            point_extend_num += points_extend.shape[0]
            storePly(os.path.join(self.partition_ori_dir, f"{partition_idx}.ply"), points, colors)  # 分别保存未拓展前 和 拓展后的点云
            storePly(os.path.join(self.partition_extend_dir, f"{partition_idx}_extend.ply"), points_extend, colors_extend)

        # 未拓展边界前：根据位置选择后的数据量会比初始的点云数量小很多，因为相机围成的边界会比实际的边界小一些，因此使用这些边界筛点云，点的数量会减少
        # 拓展边界后：因为会有许多重合的点，因此点的数量会增多
        print(f"Total ori point number: {pcd.points.shape[0]}\n", f"Total before extend point number: {point_num}\n", f"Total extend point number: {point_extend_num}\n")

        self.partition_list = partition_list

    def Visibility_based_camera_selection(self):
        # 3.基于可见性的相机选择
        # TODO: 2024.5.15 继续进行
        pass

    def Coverage_based_point_selection(self):
        # 4.基于覆盖率的点选择
        pass

