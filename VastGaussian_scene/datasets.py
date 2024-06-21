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
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, cameraList_from_camInfos_partition
from VastGaussian_scene.data_partition import ProgressiveDataPartitioning


class BigScene:
    """加载原始的大场景，用于对场景进行分割"""

    def __init__(self, args, load_iteration=None, resolution_scales=[1.0]):
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
            # scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.manhattan, args.man_trans)   # 得到一个场景的所有参数信息
            scene_info = sceneLoadTypeCallbacks["Partition"](args.source_path, args.images, args.man_trans)  # 得到一个场景的所有参数信息
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
            # self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
            #                                                                 args)
            self.train_cameras[resolution_scale] = cameraList_from_camInfos_partition(scene_info.train_cameras, args)
            # print("Loading Test Cameras")
            # self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
            #                                                                args)

        DataPartitioning = ProgressiveDataPartitioning(scene_info, self.train_cameras[resolution_scales[0]],
                                                       self.model_path, args.m_region, args.n_region, args.extend_rate, args.visible_rate)
        self.partition_data = DataPartitioning.format_data()  # 数据分区

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

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], logger=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.partition_id = args.partition_id

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            if logger:
                logger.info("Loading trained model at iteration {}".format(self.loaded_iter))
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        scene_info = sceneLoadTypeCallbacks["ColmapVast"](args.source_path, args.partition_model_path, args.partition_id, args.images, args.eval, man_trans=args.man_trans)  # 之前已经经过曼哈顿对齐，此时不用再次对齐

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, f"{self.partition_id}_point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
