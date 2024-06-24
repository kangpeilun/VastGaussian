# -*- coding: utf-8 -*-
#        Data: 2024-06-21 17:01
#     Project: VastGaussian
#   File Name: partition_utils.py
#      Author: KangPeilun
#       Email: 374774222@qq.com 
# Description:
import os.path

import scene
from utils.camera_utils import cameraList_from_camInfos_partition

def data_partition(lp):
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from scene.vastgs.data_partition import ProgressiveDataPartitioning

    # 读取整个场景的点云以及相机，同时将相机划分为train和test
    scene_info = sceneLoadTypeCallbacks["Partition"](lp.source_path, lp.images, lp.man_trans, lp.eval, lp.llffhold)  # 得到一个场景的所有参数信息
    with open(os.path.join(lp.model_path, "train_cameras.txt"), "w") as f:
        for cam in scene_info.train_cameras:
            image_name = cam.image_name
            f.write(f"{image_name}\n")

    with open(os.path.join(lp.model_path, "test_cameras.txt"), "w") as f:
        for cam in scene_info.test_cameras:
            image_name = cam.image_name
            f.write(f"{image_name}\n")

    all_cameras = cameraList_from_camInfos_partition(scene_info.train_cameras + scene_info.test_cameras, args=lp)
    DataPartitioning = ProgressiveDataPartitioning(scene_info, all_cameras, lp.model_path,
                                                   lp.m_region, lp.n_region, lp.extend_rate, lp.visible_rate)
    partition_result = DataPartitioning.partition_scene

    # 保存每个partition的图片名称到txt文件
    client = 0
    partition_id_list = []
    for partition in partition_result:
        partition_id_list.append(partition.partition_id)
        camera_info = partition.cameras
        image_name_list = [camera_info[i].camera.image_name + '.jpg' for i in range(len(camera_info))]
        txt_file = f"{lp.model_path}/partition_point_cloud/visible/{partition.partition_id}_camera.txt"
        # 打开一个文件用于写入，如果文件不存在则会被创建
        with open(txt_file, 'w') as file:
            # 遍历列表中的每个元素
            for item in image_name_list:
                # 将每个元素写入文件，每个元素占一行
                file.write(f"{item}\n")
        client += 1

    return client, partition_id_list


def read_camList(path):
    camList = []
    with open(path, "r") as f:
        lines = f.readlines()
        for image_name in lines:
            camList.append(image_name.replace("\n", ""))

    return camList


if __name__ == '__main__':
    read_camList(r"E:\Pycharm\3D_Reconstruct\VastGaussian\output\train_1\train_cameras.txt")