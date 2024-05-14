# Author: Kang Peilun
# Contact: kangpeilun@nefu.edu.cn
# Project: Geo3DGS
# File: points3D_utils.py
# Time: 2/26/24 5:33 PM
# Desc:
import os.path

import numpy as np
from scipy import spatial

from scene.colmap_loader import read_points3D_text
from scene.dataset_readers import storePly

def statistical_filter(point, n=10, std_ratio=0.0001):
    k_dist = np.zeros(len(point))
    tree = spatial.KDTree(point[:, :3])
    for i in range(0, len(point)):
        nb = tree.query(point[i, :3], k=n, workers=-1)
        nb_dt = nb[0]
        k_dist[i] = np.sum(nb_dt)
    # max_distance = np.mean(k_dist) + std_ratio * np.std(k_dist)
    max_distance = np.mean(k_dist)
    idx = np.where(k_dist < max_distance)
    point = point[idx]
    return point, idx


def filter_point3D(path, n=10, std_ratio=0.0001):
    xyzs, rgbs, errors = read_points3D_text(path)
    filter_xyzs, filter_idx = statistical_filter(xyzs, n, std_ratio)
    filter_rgbs = rgbs[filter_idx]
    filter_errors = errors[filter_idx]

    save_path = os.path.join(os.path.dirname(path), "filter_points3D.txt")
    num_of_points = len(filter_idx[0])
    with open(save_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n"
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
                f"# Number of points: {num_of_points}, mean track length: 3.8556297588157729\n")
        for idx, (xyz, rgb, error) in enumerate(zip(filter_xyzs, filter_rgbs, filter_errors)):
            line = f"{idx+1} {xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]} {error[0]}\n"
            f.write(line)

    save_ply_path = os.path.join(os.path.dirname(path), "filter_points3D.ply")
    storePly(save_ply_path, filter_xyzs, filter_rgbs)


if __name__ == '__main__':
    point3D_path = r"/home/kpl/develop/Pycharm/Projects/Geo3DGS/datasets/Colmap_DFC2019_crop/JAX_068/points3D.txt"
    filter_point3D(point3D_path, n=5)