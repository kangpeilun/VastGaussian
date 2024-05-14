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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    params: np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    """
    函数内部定义了一个名为 get_center_and_diag 的内部函数，用于计算摄像机中心点和对角线的长度。该函数接受一个参数 cam_centers，表示摄像机中心点的列表。
        在函数内部，首先将 cam_centers 转换为一个水平堆叠的数组，然后计算所有摄像机中心点的平均值，得到平均摄像机中心点 avg_cam_center。
        接下来，计算每个摄像机中心点到平均摄像机中心点的距离，并取最大距离作为对角线的长度 diagonal。最后，将平均摄像机中心点和对角线的长度返回。
    在 getNerfppNorm 函数中，首先定义了一个空列表 cam_centers，用于存储所有摄像机的中心点。
    然后，通过遍历 cam_info 中的每个摄像机，计算其世界坐标系到视图坐标系的转换矩阵 W2C，并计算其逆矩阵 C2W。将 C2W 的前三行第四列（即平移向量）添加到 cam_centers 列表中。
    接下来，调用 get_center_and_diag 函数，传入 cam_centers 列表，计算得到摄像机中心点和对角线的长度。
    然后，将对角线的长度乘以1.1，得到半径 radius。
    最后，计算平移向量 translate，即将摄像机中心点移动到原点的向量。
    最终，返回一个字典，包含 translate 和 radius 两个参数。
    """
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)  # 得到平均摄像机中心点x y z三个方向各有一个均值
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)  # np.linalg.norm表示求范数
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)  # 根据相机的旋转矩阵和平移向量，将 世界坐标系 -> 相机坐标系
        C2W = np.linalg.inv(W2C)  # 相机坐标系 -> 世界坐标系
        cam_centers.append(C2W[:3, 3:4])  # 将变换矩阵中的平移向量作为相机的中心，在世界坐标系下相机的中心

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1  # 半径

    translate = -center  # 将相机

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):  # 每个相机单独处理
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]  # 获取该图片的 相机外参，将xys的像素坐标转换为世界坐标，然后找到他们的几何中心
        intr = cam_intrinsics[extr.camera_id]  # 获取该图片所使用的相机内参
        height = intr.height  # 获取该图片的宽高
        width = intr.width

        uid = intr.id  # 获取相机对应id
        R = np.transpose(qvec2rotmat(extr.qvec))  # 由四元数获取该图片的旋转矩阵，得到世界->相机坐标的旋转矩阵
        T = np.array(extr.tvec)  # 获取该图片的平移向量
        params = np.array(intr.params)

        if intr.model == "SIMPLE_PINHOLE":  # 使用SIMPLE_PINHOLE相机模型，适用于非畸变图像，它有一个焦距参数，也可以理解为fx=fy
            focal_length_x = intr.params[0]  # 相机内参
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":  # 使用PINHOLE相机模型，适用于畸变图像，它有两个焦距参数
            focal_length_x = intr.params[0]  # 获取x轴焦距
            focal_length_y = intr.params[1]  # 获取y轴焦距
            FovY = focal2fov(focal_length_y, height)  # 获取垂直视角场  视场角Fov是指在成像场景中，相机可以接收影像的角度范围，也常被称为视野范围
            FovX = focal2fov(focal_length_x, width)   # 获取水平视角场
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"  # Colmap相机模型未处理：仅支持未失真的数据集（PINHOLE或SIMPLE_PINHOLE相机）！

        image_path = os.path.join(images_folder, os.path.basename(extr.name))  # 获取该图片路径
        image_name = os.path.basename(image_path).split(".")[0]  # 获取该图片名称
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, params=params)
        cam_infos.append(cam_info)  # 存储所有图片的 相机模型id，旋转矩阵 平移向量，视角场，图片数据，图片路径，图片名，图片宽高
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    """这段代码定义了一个名为 fetchPly 的函数，用于读取并解析一个 PLY 文件，并返回一个 BasicPointCloud 对象。
    函数接受一个参数 path，表示 PLY 文件的路径。
    首先，使用 PlyData.read 方法读取指定路径的 PLY 文件，并将结果赋值给 plydata。
    然后，从 plydata 中提取名为 'vertex' 的元素，表示点云中的顶点。
    接着，从顶点元素中提取 'x'、'y'、'z' 三个属性，分别表示顶点的 x、y、z 坐标。
        使用 np.vstack 方法将这三个属性堆叠在一起，并通过转置操作得到一个 (N, 3) 的矩阵 positions，其中 N 表示顶点的数量。
    接下来，从顶点元素中提取 'red'、'green'、'blue' 三个属性，分别表示顶点的红、绿、蓝通道的颜色值。
        使用 np.vstack 方法将这三个属性堆叠在一起，并通过除以 255.0 进行归一化，得到一个 (N, 3) 的矩阵 colors，其中 N 表示顶点的数量。
    然后，从顶点元素中提取 'nx'、'ny'、'nz' 三个属性，分别表示顶点的法线向量的 x、y、z 分量。使用 np.vstack 方法将这三个属性堆叠在一起，得到一个 (N, 3) 的矩阵 normals，其中 N 表示顶点的数量。
    最后，使用 positions、colors、normals 创建一个 BasicPointCloud 对象，并将其作为结果返回。
    """
    plydata = PlyData.read(path)
    vertices = plydata['vertex']  # 提取点云的顶点
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T  # 将x,y,z这三个坐标属性堆叠在一起
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0  # 将R,G,B三个颜色属性堆叠在一起，并除以255进行归一化
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T  # 提取顶点的三个法向量属性，并堆叠在一起
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")   # 相机外参文件
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")  # 相机内参文件
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)  # 读取相机外参
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)  # 读取相机内参
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))  # 存储所有图片的 相机模型id，旋转矩阵 平移向量，视角场，图片数据，图片路径，图片名，图片宽高
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)  # 根据图片名称对 list进行排序

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos  # 得到训练图片的相机参数
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)  # 使用找到在世界坐标系下相机的几何中心
    # 将3D点云数据写入 scene_info中
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")  # 将point3d.bin转换为.ply，只会在您第一次打开场景时发生。
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)  # 得到稀疏点云中，各个3D点的属性信息
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)  # 保存一个场景的所有参数信息
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["input"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1]))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points  由于该数据集没有colmap数据，因此我们从随机点开始
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes  我们在合成Blender场景的边界内创建随机点
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


# TODO: 完成RPC模型数据的加载
def readRpcSceneInfo():
    pass


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Rpc": readRpcSceneInfo,
}


if __name__ == '__main__':
    pass