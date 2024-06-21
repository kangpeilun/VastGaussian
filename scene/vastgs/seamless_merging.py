# Author: Peilun Kang
# Contact: kangpeilun@nefu.edu.cn
# License: Apache Licence
# Project: VastGaussian
# File: seamless_merging.py
# Time: 5/15/24 2:31 PM
# Des: 无缝合并
import os.path
import pickle
import numpy as np
import torch
from plyfile import PlyData, PlyElement

def load_ply(path):
    plydata = PlyData.read(path)
    max_sh_degree = 3
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
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

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

    return xyz, features_dc, features_extra, opacities, scales, rots


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


def save_ply(path, xyz, f_dc, f_rest, opacities, scale, rotation):
    normals = np.zeros_like(xyz)

    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(f_dc.shape[1] * f_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(f_rest.shape[1] * f_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scale.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))
    dtype_full = [(attribute, 'f4') for attribute in l]

    f_dc = torch.tensor(f_dc).transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = torch.tensor(f_rest).transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def extract_point_cloud(points, bbox):
    """根据camera的边界从初始点云中筛选对应partition的点云"""
    mask = (points[:, 0] >= bbox[0]) & (points[:, 0] <= bbox[1]) & (
            points[:, 1] >= bbox[2]) & (points[:, 1] <= bbox[3]) & (
                   points[:, 2] >= bbox[4]) & (points[:, 2] <= bbox[5])  # 筛选在范围内的点云，得到对应的mask
    return mask


def seamless_merge(model_path, partition_point_cloud_dir):
    save_merge_dir = os.path.join(partition_point_cloud_dir, "point_cloud.ply")

    # 加载partition数据
    with open(os.path.join(model_path, "partition_data.pkl"), "rb") as f:
        partition_scene = pickle.load(f)

    # 遍历所有partition点云
    xyz_list = []
    features_dc_list = []
    features_extra_list = []
    opacities_list = []
    scales_list = []
    rots_list = []
    for partition in partition_scene:
        point_cloud_path = os.path.join(partition_point_cloud_dir, f"{partition.partition_id}_point_cloud.ply")
        xyz, features_dc, features_extra, opacities, scales, rots = load_ply(point_cloud_path)
        extend_camera_bbox = partition.extend_camera_bbox  # 原始相机包围盒
        extend_point_bbox = partition.extend_point_bbox  # 原始点云包围盒
        point_select_bbox = [extend_camera_bbox[0], extend_camera_bbox[1],  # [x_min, x_max, y_min, y_max, z_min, z_max]
                             -np.inf, np.inf,
                             # 考虑原始点云的包围盒的y轴范围作为还原的范围，因为在partition时，没有考虑y轴方向
                             extend_camera_bbox[2], extend_camera_bbox[3]]
        mask = extract_point_cloud(xyz, point_select_bbox)
        xyz_list.append(xyz[mask])
        features_dc_list.append(features_dc[mask])
        features_extra_list.append(features_extra[mask])
        opacities_list.append(opacities[mask])
        scales_list.append(scales[mask])
        rots_list.append(rots[mask])

    points = np.concatenate(xyz_list, axis=0)
    features_dc_list = np.concatenate(features_dc_list, axis=0)
    features_extra_list = np.concatenate(features_extra_list, axis=0)
    opacities_list = np.concatenate(opacities_list, axis=0)
    scales_list = np.concatenate(scales_list, axis=0)
    rots_list = np.concatenate(rots_list, axis=0)

    # 因为使用拓展后的边界进行组合，因此可能会有一些重合的点，因此去重
    points, mask = np.unique(points, axis=0, return_index=True)
    features_dc_list = features_dc_list[mask]
    features_extra_list = features_extra_list[mask]
    opacities_list = opacities_list[mask]
    scales_list = scales_list[mask]
    rots_list = rots_list[mask]

    save_ply(save_merge_dir, points, features_dc_list, features_extra_list, opacities_list, scales_list, rots_list)


if __name__ == '__main__':
    seamless_merge("/home/kpl/develop/Pycharm/Projects/VastGaussian/output/train",
                   "/home/kpl/develop/Pycharm/Projects/VastGaussian/output/train/point_cloud/iteration_20")
