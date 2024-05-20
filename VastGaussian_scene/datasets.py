# Author: Peilun Kang
# Contact: kangpeilun@nefu.edu.cn
# License: Apache Licence
# Project: VastGaussian
# File: datasets.py
# Time: 5/17/24 1:19 PM
# Des:

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from VastGaussian_scene.data_partition import ProgressiveDataPartitioning


class BigScene:
    """加载原始的大场景，用于对场景进行分割"""

    def __init__(self, args: ModelParams, load_iteration=None,
                 resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)  # 得到一个场景的所有参数信息
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())  # 将3D点云数据point3D.ply重写入 input.ply中
            json_cams = []
            camlist = []  # 保存相机的内外参 list
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)


        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)

        # TODO：实现数据预处理
        DataPartitioning = ProgressiveDataPartitioning(scene_info, self.train_cameras[resolution_scales[0]],
                                                       self.model_path)
        self.partition_data = DataPartitioning.format_data()  # 数据分区
        # self.train_cameras[resolution_scale] = format_data['1_1']['cameras']  # 通过这样的方式将数据取出来
        # self.gaussians.create_from_pcd(format_data['1_1']['point_cloud'], self.cameras_extent)

        # TODO: 编写代码实现m*n个partition并行训练

        # if self.loaded_iter:
        #     self.gaussians.load_ply(os.path.join(self.model_path,
        #                                          "point_cloud",
        #                                          "iteration_" + str(self.loaded_iter),
        #                                          "point_cloud.ply"))  # 表示从断点中加载模型
        # else:
        #     self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)  # 对高斯模型的参数进行初始化, 主要使用3D点的xyz坐标，rgb值进行高斯模型的初始化

    # def save(self, iteration):
    #     point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
    #     self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    #
    # def getTrainCameras(self, scale=1.0):
    #     return self.train_cameras[scale]
    #
    # def getTestCameras(self, scale=1.0):
    #     return self.test_cameras[scale]


class PartitionScene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, partition_id, partition_data, cameras_extent,
                 shuffle=True,
                 resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.gaussians = gaussians
        self.train_cameras = partition_data[0]
        self.partition_id = partition_id
        self.cameras_extent = cameras_extent

        if shuffle:
            random.shuffle(self.train_cameras)  # Multi-res consistent random shuffling

        self.gaussians.create_from_pcd(partition_data[1], cameras_extent)  # 对高斯模型的参数进行初始化, 主要使用3D点的xyz坐标，rgb值进行高斯模型的初始化

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, f"point_cloud_{self.partition_id}.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras

    def getTestCameras(self, scale=1.0):
        return self.test_cameras
