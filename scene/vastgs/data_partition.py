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
import copy
import os
import numpy as np
from typing import NamedTuple
import pickle
import math

from scene.dataset_readers import CameraInfo, storePly
from utils.graphics_utils import BasicPointCloud
from scene.vastgs.graham_scan import run_graham_scan
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CameraPose(NamedTuple):
    camera: CameraInfo
    pose: np.array  # [x, y, z] 坐标


class CameraPartition(NamedTuple):
    partition_id: str  # 部分的字符编号
    cameras: list  # 该部分对应的所有相机 CameraPose
    point_cloud: BasicPointCloud  # 该部分对应的点云
    ori_camera_bbox: list  # 边界拓展前相机围成的相机 边界坐标 [x_min, x_max, z_min, z_max]，同时方便根据原始的边框对最后训练出来的点云进行裁减，获取原始的点云范围
    extend_camera_bbox: list  # 按照extend_rate拓展后 相机围成的边界坐标
    extend_rate: float  # 边界拓展的比例 默认0.2

    ori_point_bbox: list  # 由拓展前相机边界筛选出来的point，这些点的边界  [x_min, x_max, y_min, y_max, z_min, z_max]
    extend_point_bbox: list  # 由拓展后相机边界筛选出来的point，这些点的边界


class ProgressiveDataPartitioning:
    # 渐进数据分区
    def __init__(self, scene_info, train_cameras, model_path, m_region=2, n_region=4, extend_rate=0.2,
                 visible_rate=0.25):
        self.partition_scene = None
        self.pcd = scene_info.point_cloud
        # print(f"self.pcd={self.pcd}")
        self.model_path = model_path  # 存放模型位置
        self.partition_dir = os.path.join(model_path, "partition_point_cloud")
        self.partition_ori_dir = os.path.join(self.partition_dir, "ori")
        self.partition_extend_dir = os.path.join(self.partition_dir, "extend")
        self.partition_visible_dir = os.path.join(self.partition_dir, "visible")
        self.save_partition_data_dir = os.path.join(self.model_path, "partition_data.pkl")
        self.m_region = m_region
        self.n_region = n_region
        self.extend_rate = extend_rate
        self.visible_rate = visible_rate

        if not os.path.exists(self.partition_ori_dir): os.makedirs(self.partition_ori_dir)  # 创建存放分块后 拓展前 点云的文件夹
        if not os.path.exists(self.partition_extend_dir): os.makedirs(self.partition_extend_dir)  # 创建存放分块后 拓展后 点云的文件夹
        if not os.path.exists(self.partition_visible_dir): os.makedirs(
            self.partition_visible_dir)  # 创建存放分块后 可见性相机选择后 点云的文件夹
        self.fig, self.ax = self.draw_pcd(self.pcd, train_cameras)
        self.run_DataPartition(train_cameras)

    def draw_pcd(self, pcd, train_cameras):        
        x_coords = pcd.points[:, 0]
        z_coords = pcd.points[:, 2]
        fig, ax = plt.subplots()
        ax.scatter(x_coords, z_coords, c=(pcd.colors), s=1)
        ax.title.set_text('Plot of 2D Points')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Z-axis')
        fig.tight_layout()
        fig.savefig(os.path.join(self.model_path, 'pcd.png'),dpi=200)
        x_coords = np.array([cam.camera_center[0].item() for cam in train_cameras])
        z_coords = np.array([cam.camera_center[2].item() for cam in train_cameras])
        ax.scatter(x_coords, z_coords, color='red', s=1)
        fig.savefig(os.path.join(self.model_path, 'camera_on_pcd.png'),dpi=200)
        return fig, ax
        
    def draw_partition(self, partition_list):
        for partition in partition_list:
            ori_bbox = partition.ori_camera_bbox
            extend_bbox = partition.extend_camera_bbox
            x_min, x_max, z_min, z_max = ori_bbox
            ex_x_min, ex_x_max, ex_z_min, ex_z_max = extend_bbox
            rect_ori = patches.Rectangle((x_min, z_min), x_max - x_min, z_max - z_min, linewidth=1, edgecolor='blue',
                                         facecolor='none')
            rect_ext = patches.Rectangle((ex_x_min, ex_z_min), ex_x_max-ex_x_min, ex_z_max-ex_z_min, linewidth=1, edgecolor='y', facecolor='none')
            self.ax.add_patch(rect_ori)
            self.ax.text(x=rect_ori.get_x(), y=rect_ori.get_y(), s=f"{partition.partition_id}", color='black', fontsize=12)
            self.ax.add_patch(rect_ext)
        self.fig.savefig(os.path.join(self.model_path, f'regions.png'),dpi=200)
        return
        
    def run_DataPartition(self, train_cameras):
        if not os.path.exists(self.save_partition_data_dir):
            partition_dict = self.Camera_position_based_region_division(train_cameras)
            partition_dict, refined_ori_bbox = self.refine_ori_bbox(partition_dict)
            # partition_dict, refined_ori_bbox = self.refine_ori_bbox_average(partition_dict)
            partition_list = self.Position_based_data_selection(partition_dict, refined_ori_bbox)
            self.draw_partition(partition_list)
            self.partition_scene = self.Visibility_based_camera_selection(partition_list)  # 输出经过可见性筛选后的场景 包括相机和点云
            self.save_partition_data()
        else:
            self.partition_scene = self.load_partition_data()


    def save_partition_data(self):
        """将partition后的数据序列化保存起来, 方便下次加载"""
        with open(self.save_partition_data_dir, 'wb') as f:
            pickle.dump(self.partition_scene, f)

    def load_partition_data(self):
        """加载partition后的数据"""
        with open(self.save_partition_data_dir, 'rb') as f:
            partition_scene = pickle.load(f)
        return partition_scene

    def refine_ori_bbox_average(self, partition_dict):
        """修正原始的bbox，使得边界变得无缝，方便后续进行无缝合并
        取相邻两个相机边界框的平均值作为无缝合并的边界
        """
        bbox_with_id = {}
        # 1.获取每个分区的相机边界
        for partition_idx, cameras in partition_dict.items():
            # TODO: 需要修改，origin边界用分区时的边界，不能使用相机的位置作为边界，否则无法做到无缝合并
            camera_list = cameras["camera_list"]
            min_x, max_x = min(camera.pose[0] for camera in camera_list), max(
                camera.pose[0] for camera in camera_list)  # min_x, max_x表示相机围成的区域的x轴方向的长度
            min_z, max_z = min(camera.pose[2] for camera in camera_list), max(camera.pose[2] for camera in camera_list)
            ori_camera_bbox = [min_x, max_x, min_z, max_z]
            bbox_with_id[partition_idx] = ori_camera_bbox

        # 2.按照先x轴对相机的边界进行修正，以相邻两个分区边界坐标的平均值作为公共边界
        for m in range(1, self.m_region+1):
            for n in range(1, self.n_region+1):
                if n+1 == self.n_region+1:
                    # 如果是最后一块就退出
                    break
                partition_idx_1 = str(m) + '_' + str(n)    # 左边块
                min_x_1, max_x_1, min_z_1, max_z_1 = bbox_with_id[partition_idx_1]
                partition_idx_2 = str(m) + '_' + str(n+1)  # 右边快
                min_x_2, max_x_2, min_z_2, max_z_2 = bbox_with_id[partition_idx_2]
                mid_z = (max_z_1 + min_z_2) / 2
                bbox_with_id[partition_idx_1] = [min_x_1, max_x_1, min_z_1, mid_z]
                bbox_with_id[partition_idx_2] = [min_x_2, max_x_2, mid_z, max_z_2]

        # 3.再按z轴的顺序对相机边界进行修正，因为是按照x轴进行修正的，因此需要按照x轴对左右两边的块进行均分
        # 先找到左右两边中，左边分区的最大x_max, 右边分区的最小x_min
        for m in range(1, self.m_region + 1):
            if m + 1 == self.m_region + 1:
                # 如果是最后一块就退出
                break
            max_x_left = -np.inf
            min_x_right = np.inf
            for n in range(1, self.n_region+1):   # 左边分区
                partition_idx = str(m) + '_' + str(n)
                min_x, max_x, min_z, max_z = bbox_with_id[partition_idx]
                if max_x > max_x_left: max_x_left = max_x

            for n in range(1, self.n_region+1):   # 右边分区
                partition_idx = str(m+1) + '_' + str(n)
                min_x, max_x, min_z, max_z = bbox_with_id[partition_idx]
                if min_x < min_x_right: min_x_right = min_x

            # 修正左右两边分区的边界
            for n in range(1, self.n_region+1):   # 左边分区
                partition_idx = str(m) + '_' + str(n)
                min_x, max_x, min_z, max_z = bbox_with_id[partition_idx]
                mid_x = (max_x_left + min_x_right) / 2
                bbox_with_id[partition_idx] = [min_x, mid_x, min_z, max_z]

            for n in range(1, self.n_region + 1):  # 右边分区
                partition_idx = str(m+1) + '_' + str(n)
                min_x, max_x, min_z, max_z = bbox_with_id[partition_idx]
                mid_x = (max_x_left + min_x_right) / 2
                bbox_with_id[partition_idx] = [mid_x, max_x, min_z, max_z]

        new_partition_dict = {f"{partition_id}": cameras["camera_list"] for partition_id, cameras in
                              partition_dict.items()}
        return new_partition_dict, bbox_with_id

    def refine_ori_bbox(self, partition_dict):
        """将连续的相机坐标作为无缝分块的边界"""
        bbox_with_id = {}
        for partition_idx, cameras in partition_dict.items():
            # TODO: 需要修改，origin边界用分区时的边界，不能使用相机的位置作为边界，否则无法做到无缝合并
            camera_list = cameras["camera_list"]
            min_x, max_x = min(camera.pose[0] for camera in camera_list), max(
                camera.pose[0] for camera in camera_list)  # min_x, max_x表示相机围成的区域的x轴方向的长度
            min_z, max_z = min(camera.pose[2] for camera in camera_list), max(camera.pose[2] for camera in camera_list)
            ori_camera_bbox = [min_x, max_x, min_z, max_z]
            bbox_with_id[partition_idx] = ori_camera_bbox

        # 2.按照z轴对相机的边界进行修正
        for m in range(1, self.m_region+1):
            for n in range(1, self.n_region+1):
                if n+1 == self.n_region+1:
                    break
                partition_idx_1 = str(m) + '_' + str(n+1)  # 上边块
                min_x_1, max_x_1, min_z_1, max_z_1 = bbox_with_id[partition_idx_1]
                partition_idx_2 = str(m) + '_' + str(n)  # 下边块
                min_x_2, max_x_2, min_z_2, max_z_2 = bbox_with_id[partition_idx_2]
                mid_x, mid_y, mid_z = partition_dict[partition_idx_2]["z_mid_camera"].pose
                bbox_with_id[partition_idx_1] = [min_x_1, max_x_1, mid_z, max_z_1]
                bbox_with_id[partition_idx_2] = [min_x_2, max_x_2, min_z_2, mid_z]

        # 3.按照x轴对相机的边界进行修正
        for n in range(1, self.n_region + 1):
            for m in range(1, self.m_region + 1):
                if m + 1 == self.m_region + 1:
                    break
                partition_idx_1 = str(m) + '_' + str(n)  # 左边块
                min_x_1, max_x_1, min_z_1, max_z_1 = bbox_with_id[partition_idx_1]
                partition_idx_2 = str(m+1) + '_' + str(n)  # 右边块
                min_x_2, max_x_2, min_z_2, max_z_2 = bbox_with_id[partition_idx_2]
                mid_x, mid_y, mid_z = partition_dict[partition_idx_1]["x_mid_camera"].pose
                bbox_with_id[partition_idx_1] = [min_x_1, mid_x, min_z_1, max_z_1]
                bbox_with_id[partition_idx_2] = [mid_x, max_x_2, min_z_2, max_z_2]

        new_partition_dict = {f"{partition_id}": cameras["camera_list"] for partition_id, cameras in partition_dict.items()}
        return new_partition_dict, bbox_with_id

    def Camera_position_based_region_division(self, train_cameras):
        """1.基于相机位置的区域划分
        思路: 1.首先将整个场景的相机坐标投影到以xz轴组成的平面上
             2.按照x轴方向, 将所有的相机分成m部分
             3.按照z轴方向, 将每一部分分成n部分 (默认将整个区域分成2*4=8个部分),同时保证mxn个部分中的相机数量的均衡
             4.返回每个部分的边界坐标，以及每个部分对应的相机
        """
        m, n = self.m_region, self.n_region    # m=2, n=4
        CameraPose_list = []
        camera_centers = []
        for idx, camera in enumerate(train_cameras):
            pose = np.array(camera.camera_center.cpu())
            camera_centers.append(pose)
            CameraPose_list.append(
                CameraPose(camera=camera, pose=pose))  # 世界坐标系下相机的中心坐标

        # 保存相机坐标，用于可视化相机位置
        storePly(os.path.join(self.partition_dir, 'camera_centers.ply'), np.array(camera_centers), np.zeros_like(np.array(camera_centers)))

        # 2.沿着x轴将相机分成m部分
        m_partition_dict = {}
        total_camera = len(CameraPose_list)  # 获取相机总数
        num_of_camera_per_m_partition = total_camera // m  # m个部分，每部分相机数量
        sorted_CameraPose_by_x_list = sorted(CameraPose_list, key=lambda x: x.pose[0])  # 按照x轴坐标排序
        # print(sorted_CameraPose_by_x_list)
        for i in range(m):  # 按照x轴将所有相机分成m部分
            m_partition_dict[str(i + 1)] = {"camera_list": sorted_CameraPose_by_x_list[
                                           i * num_of_camera_per_m_partition:(i + 1) * num_of_camera_per_m_partition]}
            if i != m-1:
                m_partition_dict[str(i + 1)].update({"x_mid_camera": sorted_CameraPose_by_x_list[(i + 1) * num_of_camera_per_m_partition-1]})  # 将左边块的相机作为无缝衔接的边界
            else:
                m_partition_dict[str(i + 1)].update({"x_mid_camera": None})  # 最后一块不需要mid_camera
        if total_camera % m != 0:  # 如果相机数量不是m的整数倍，则将余下的相机直接添加到最后一部分
            m_partition_dict[str(m)]["camera_list"].extend(sorted_CameraPose_by_x_list[m * num_of_camera_per_m_partition:])

        # 3.沿着z轴将相机分成n部分
        partition_dict = {}  # 保存mxn每个部分的相机数量
        for partition_idx, cameras in m_partition_dict.items():
            partition_total_camera = len(cameras["camera_list"])  # m个部分，每部分相机数量
            num_of_camera_per_n_partition = partition_total_camera // n  # n个部分，每部分相机数量
            sorted_CameraPose_by_z_list = sorted(cameras["camera_list"], key=lambda x: x.pose[2])  # 按照z轴坐标排序
            for i in range(n):  # 按照z轴将所有相机分成n部分
                partition_dict[f"{partition_idx}_{i + 1}"] = {"camera_list": sorted_CameraPose_by_z_list[
                                                             i * num_of_camera_per_n_partition:(i + 1) * num_of_camera_per_n_partition]}
                if i != n-1:
                    partition_dict[f"{partition_idx}_{i + 1}"].update({"x_mid_camera": cameras["x_mid_camera"]})
                    partition_dict[f"{partition_idx}_{i + 1}"].update({"z_mid_camera": sorted_CameraPose_by_z_list[(i + 1) * num_of_camera_per_n_partition - 1]})
                else:
                    partition_dict[f"{partition_idx}_{i + 1}"].update({"x_mid_camera": cameras["x_mid_camera"]})
                    partition_dict[f"{partition_idx}_{i + 1}"].update({"z_mid_camera": None})  # 最后一块不需要mid_camera
            if partition_total_camera % n != 0:  # 如果相机数量不是n的整数倍，则将余下的相机直接添加到最后一部分
                partition_dict[f"{partition_idx}_{n}"]["camera_list"].extend(
                    sorted_CameraPose_by_z_list[n * num_of_camera_per_n_partition:])

        return partition_dict

    def extract_point_cloud(self, pcd, bbox):
        """根据camera的边界从初始点云中筛选对应partition的点云"""
        mask = (pcd.points[:, 0] >= bbox[0]) & (pcd.points[:, 0] <= bbox[1]) & (
                pcd.points[:, 2] >= bbox[2]) & (pcd.points[:, 2] <= bbox[3])  # 筛选在范围内的点云，得到对应的mask
        points = pcd.points[mask]
        colors = pcd.colors[mask]
        normals = pcd.normals[mask]
        return points, colors, normals

    def get_point_range(self, points):
        """获取当前点云的x y z边界"""
        x_list = points[:, 0]
        y_list = points[:, 1]
        z_list = points[:, 2]
        # print(points.shape)
        return [min(x_list), max(x_list),
                min(y_list), max(y_list),
                min(z_list), max(z_list)]

    def Position_based_data_selection(self, partition_dict, refined_ori_bbox):
        """
        2.基于位置的数据选择
        思路: 1.计算每个partition的x z边界
             2.然后按照extend_rate将每个partition的边界坐标扩展, 得到新的边界坐标 [x_min, x_max, z_min, z_max]
             3.根据extend后的边界坐标, 获取该部分对应的点云
        问题: 有可能根据相机确定边界框后, 仍存在一些比较好的点云没有被选中的情况, 因此extend_rate是一个超参数, 需要根据实际情况调整
        :return partition_list: 每个部分对应的点云，所有相机，边界
        """
        # 计算每个部分的拓展后的边界坐标，以及该部分对应的点云
        pcd = self.pcd
        partition_list = []
        point_num = 0
        point_extend_num = 0
        for partition_idx, camera_list in partition_dict.items():
            min_x, max_x, min_z, max_z = refined_ori_bbox[partition_idx]
            ori_camera_bbox = [min_x, max_x, min_z, max_z]
            extend_camera_bbox = [min_x - self.extend_rate * (max_x - min_x),
                                  max_x + self.extend_rate * (max_x - min_x),
                                  min_z - self.extend_rate * (max_z - min_z),
                                  max_z + self.extend_rate * (max_z - min_z)]
            print("Partition", partition_idx, "ori_camera_bbox", ori_camera_bbox, "\textend_camera_bbox", extend_camera_bbox)
            ori_camera_centers = []
            for camera_pose in camera_list:
                ori_camera_centers.append(camera_pose.pose)

            # 保存ori相机位置
            storePly(os.path.join(self.partition_ori_dir, f'{partition_idx}_camera_centers.ply'),
                     np.array(ori_camera_centers),
                     np.zeros_like(np.array(ori_camera_centers)))

            # TODO: 需要根据拓展后的边界重新添加相机
            new_camera_list = []
            extend_camera_centers = []
            for id, camera_list in partition_dict.items():
                for camera_pose in camera_list:
                    if extend_camera_bbox[0] <= camera_pose.pose[0] <= extend_camera_bbox[1] and extend_camera_bbox[2] <= camera_pose.pose[2] <= extend_camera_bbox[3]:
                        extend_camera_centers.append(camera_pose.pose)
                        new_camera_list.append(camera_pose)

            # 保存extend后新添加的相机位置
            storePly(os.path.join(self.partition_extend_dir, f'{partition_idx}_camera_centers.ply'),
                     np.array(extend_camera_centers),
                     np.zeros_like(np.array(extend_camera_centers)))

            # 获取该部分对应的点云
            points, colors, normals = self.extract_point_cloud(pcd, ori_camera_bbox)  # 分别提取原始边界内的点云，和拓展边界后的点云
            points_extend, colors_extend, normals_extend = self.extract_point_cloud(pcd, extend_camera_bbox)
            # 论文中说点云围成的边界框的高度选取为最高点到地平面的距离，但在本实现中，因为不确定地平面位置，(可视化中第平面不用坐标轴xz重合)
            # 因此使用整个点云围成的框作为空域感知的边界框
            partition_list.append(CameraPartition(partition_id=partition_idx, cameras=new_camera_list,
                                                  point_cloud=BasicPointCloud(points_extend, colors_extend, normals_extend),
                                                  ori_camera_bbox=ori_camera_bbox,
                                                  extend_camera_bbox=extend_camera_bbox,
                                                  extend_rate=self.extend_rate,
                                                  ori_point_bbox=self.get_point_range(points),
                                                  extend_point_bbox=self.get_point_range(points_extend),
                                                  ))

            point_num += points.shape[0]
            point_extend_num += points_extend.shape[0]
            storePly(os.path.join(self.partition_ori_dir, f"{partition_idx}.ply"), points, colors)  # 分别保存未拓展前 和 拓展后的点云
            storePly(os.path.join(self.partition_extend_dir, f"{partition_idx}_extend.ply"), points_extend,
                     colors_extend)

        # 未拓展边界前：根据位置选择后的数据量会比初始的点云数量小很多，因为相机围成的边界会比实际的边界小一些，因此使用这些边界筛点云，点的数量会减少
        # 拓展边界后：因为会有许多重合的点，因此点的数量会增多
        print(f"Total ori point number: {pcd.points.shape[0]}\n", f"Total before extend point number: {point_num}\n",
              f"Total extend point number: {point_extend_num}\n")

        return partition_list


    def get_8_corner_points(self, bbox):
        """根据点云的边界框，生成8个角点的坐标
        :param bbox: [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        return {
            "minx_miny_minz": [x_min, y_min, z_min],  # 1
            "minx_miny_maxz": [x_min, y_min, z_max],  # 2
            "minx_maxy_minz": [x_min, y_max, z_min],  # 3
            "minx_maxy_maxz": [x_min, y_max, z_max],  # 4
            "maxx_miny_minz": [x_max, y_min, z_min],  # 5
            "maxx_miny_maxz": [x_max, y_min, z_max],  # 6
            "maxx_maxy_minz": [x_max, y_max, z_min],  # 7
            "maxx_maxy_maxz": [x_max, y_max, z_max]   # 8
        }


    def point_in_image(self, camera, points):
        """使用投影矩阵将角点投影到二维平面"""
        # 获取点在图像平面的坐标
        R = camera.R
        T = camera.T
        w2c = np.eye(4)
        w2c[:3, :3] = np.transpose(R)
        w2c[:3, 3] = T
        fx = camera.image_width / (2 * math.tan(camera.FoVx / 2))
        fy = camera.image_height / (2 * math.tan(camera.FoVy / 2))

        intrinsic_matrix = np.array([
            [fx, 0, camera.image_height // 2],
            [0, fy, camera.image_width // 2],
            [0, 0, 1]
        ])

        points_camera = np.dot(w2c[:3, :3], points.T) + w2c[:3, 3:].reshape(3, 1)  # [3, n]
        points_camera = points_camera.T  # [n, 3]  [1, 3]
        points_camera = points_camera[np.where(points_camera[:, 2] > 0)]  # [n, 3]  这里需要根据z轴过滤一下点
        points_image = np.dot(intrinsic_matrix, points_camera.T)  # [3, n]
        points_image = points_image[:2, :] / points_image[2, :]  # [2, n]
        points_image = points_image.T  # [n, 2]

        mask = np.where(np.logical_and.reduce((
            points_image[:, 0] >= 0,
            points_image[:, 0] < camera.image_height,
            points_image[:, 1] >= 0,
            points_image[:, 1] < camera.image_width
        )))[0]

        return points_image, points_image[mask], mask


    def Visibility_based_camera_selection(self, partition_list):
        """3.基于可见性的相机选择 和 基于覆盖率的点选择
        思路：引入空域感知的能见度计算
            1.假设当前部分为i，选择j部分中的相机，
            2.将i部分边界框投影到j中的相机中，得到投影区域的面积（边界框只取地上的部分，并且可以分成拓展前和拓展后两种边界框讨论）
            3.计算投影区域面积与图像像素面积的比值，作为能见度
            4.将j中能见度大于阈值的相机s加入i中
            5.将j中所有可以投影到相机s的点云加入到i中
        :param visible_rate: 能见度阈值 默认为0.25 同论文
        """
        # 复制一份新的变量，用于添加可视相机后的每个部分的所有相机
        # 防止相机和点云被重复添加
        add_visible_camera_partition_list = copy.deepcopy(partition_list)
        client = 0
        for idx, partition_i in enumerate(partition_list):  # 第i个partition
            new_points = []  # 提前创建空的数组 用于保存新增的点
            new_colors = []
            new_normals = []

            pcd_i = partition_i.point_cloud
            partition_id_i = partition_i.partition_id  # 获取当前partition的编号
            # 获取当前partition中点云围成的边界框的8角坐标
            partition_ori_point_bbox = partition_i.ori_point_bbox
            partition_extend_point_bbox = partition_i.extend_point_bbox
            ori_8_corner_points = self.get_8_corner_points(partition_ori_point_bbox)  # 获取点云围成的边界的8个角点的坐标
            extent_8_corner_points = self.get_8_corner_points(partition_extend_point_bbox)

            corner_points = []
            for point in extent_8_corner_points.values():
                corner_points.append(point)
            storePly(os.path.join(self.partition_extend_dir, f'{partition_id_i}_corner_points.ply'),
                     np.array(corner_points),
                     np.zeros_like(np.array(corner_points)))

            total_partition_camera_count = 0  # 当前partition中的相机数量
            for partition_j in partition_list:  # 第j个partiiton
                partition_id_j = partition_j.partition_id  # 获取当前partition的编号
                if partition_id_i == partition_id_j: continue  # 如果当前partition与之前相同，则跳过
                print(f"Now processing partition i:{partition_id_i} and j:{partition_id_j}")
                # 获取当前partition中的点云
                pcd_j = partition_j.point_cloud

                append_camera_count = 0  # 用于记录第j个parition被添加了个新相机
                # 依次获取第j个partition中每个相机的投影矩阵
                # Visibility_based_camera_selection
                for cameras_pose in partition_j.cameras:
                    camera = cameras_pose.camera  # 获取当前相机
                    # 将i区域的点云投影到相机平面
                    # 3D points distributed on the object surface
                    # _, points_in_image, _ = self.point_in_image(camera, pcd_i.points)
                    # if not len(points_in_image) > 3: continue

                    # 将i部分的point_cloud边界框投影到j的当前相机中
                    # Visibility_based_camera_selection
                    # airspace-aware visibility
                    proj_8_corner_points = {}
                    for key, point in extent_8_corner_points.items():
                        points_in_image, _, _ = self.point_in_image(camera, np.array([point]))
                        if len(points_in_image) == 0: continue
                        proj_8_corner_points[key] = points_in_image[0]

                    # 基于覆盖率的点选择
                    # i部分中点云边界框投影在j部分当前图像中的面积与当前图像面积的比值
                    if not len(list(proj_8_corner_points.values())) > 3: continue
                    pkg = run_graham_scan(list(proj_8_corner_points.values()), camera.image_width, camera.image_height)
                    # pkg = run_graham_scan(points_in_image, camera.image_width, camera.image_height)
                    if pkg["intersection_rate"] >= self.visible_rate:
                        collect_names = [camera_pose.camera.image_name for camera_pose in add_visible_camera_partition_list[idx].cameras]
                        if cameras_pose.camera.image_name in collect_names:
                            # print("skip")
                            continue  # 如果相机已经存在，则不需要再重复添加
                        append_camera_count += 1
                        # print(f"Partition {idx} Append Camera {camera.image_name}")
                        # 如果空域感知比率大于阈值，则将j中的当前相机添加到i部分中
                        add_visible_camera_partition_list[idx].cameras.append(cameras_pose)
                        # 筛选在j部分中的所有点中哪些可以投影在当前图像中
                        _, _, mask = self.point_in_image(camera, pcd_j.points)  # 在原始点云上需要新增的点
                        updated_points, updated_colors, updated_normals = pcd_j.points[mask], pcd_j.colors[mask], pcd_j.normals[mask]
                        # 更新i部分的需要新增的点云，因为有许多相机可能会观察到相同的点云，因此需要对点云进行去重
                        new_points.append(updated_points)
                        new_colors.append(updated_colors)
                        new_normals.append(updated_normals)

                        with open(os.path.join(self.model_path, "graham_scan"), 'a') as f:
                            f.write(f"intersection_area:{pkg['intersection_area']}\t"
                                    f"image_area:{pkg['image_area']}\t"
                                    f"intersection_rate:{pkg['intersection_rate']}\t"
                                    f"partition_i:{partition_id_i}\t"
                                    f"partition_j:{partition_id_j}\t"
                                    f"append_camera_id:{camera.image_name}\t"
                                    f"append_camera_count:{append_camera_count}\n")
                total_partition_camera_count += append_camera_count

            with open(os.path.join(self.model_path, "partition_cameras"), 'a') as f:
                f.write(f"partition_id:{partition_id_i}\t"
                        f"total_append_camera_count:{total_partition_camera_count}\t"
                        f"total_camera:{len(add_visible_camera_partition_list[idx].cameras)}\n")

            camera_centers = []
            for camera_pose in add_visible_camera_partition_list[idx].cameras:
                camera_centers.append(camera_pose.pose)

            # 保存相机坐标，用于可视化相机位置
            storePly(os.path.join(self.partition_visible_dir, f'{partition_id_i}_camera_centers.ply'), np.array(camera_centers),
                     np.zeros_like(np.array(camera_centers)))

            # 点云去重
            point_cloud = add_visible_camera_partition_list[idx].point_cloud
            new_points.append(point_cloud.points)
            new_colors.append(point_cloud.colors)
            new_normals.append(point_cloud.normals)
            new_points = np.concatenate(new_points, axis=0)
            new_colors = np.concatenate(new_colors, axis=0)
            new_normals = np.concatenate(new_normals, axis=0)

            new_points, mask = np.unique(new_points, return_index=True, axis=0)
            new_colors = new_colors[mask]
            new_normals = new_normals[mask]

            # 当第j部分所有相机都筛选完之后，更新最终的点云
            add_visible_camera_partition_list[idx] = add_visible_camera_partition_list[idx]._replace(
                point_cloud=BasicPointCloud(points=new_points, colors=new_colors,
                                            normals=new_normals))  # 更新点云，新增的点云有许多重复的点，需要在后面剔除掉
            # store_path = os.path.join(self.partition_visible_dir, str(client))
            # if not os.path.exists(store_path): os.makedirs(store_path)
            # storePly(os.path.join(self.partition_visible_dir, str(client), f"visible.ply"), new_points, new_colors)  # 保存可见性选择后每个partition的点云
            # client += 1
            storePly(os.path.join(self.partition_visible_dir, f"{partition_id_i}_visible.ply"), new_points,
                     new_colors)  # 保存可见性选择后每个partition的点云

        return add_visible_camera_partition_list

    # def format_data(self):
    #     """对经过数据分区后的数据的格式进行规范，使得和一开始的3DGS的数据格式一致
    #     思路：1.输出每个partition的cameras
    #          2.输出每个partition的点云
    #          3.计算每个partition中相机的尺寸, 目前先使用通过整个场景计算出来的尺寸
    #     """
    #     format_data = []
    #     for partition in self.partition_scene:
    #         partition_id = partition.partition_id
    #         point_cloud = partition.point_cloud
    #         cameras = [CameraPose.camera for CameraPose in partition.cameras]
    #         cameras_extent = getNerfppNorm_partition(cameras)["radius"]  # 对每个分块后的区域都分别计算一次cameras_extent
    #         format_data.append([cameras, point_cloud, cameras_extent])
    #
    #     return format_data
