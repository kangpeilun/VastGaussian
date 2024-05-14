# -*- coding: utf-8 -*- 
# @Author: Kang Peilun
# @Email: 374774222@qq.com
# @File: process_s2p_cloud.py
# @Project: satnerf
# @Time: 2023/12/19 下午12:16
# @Des:
import os
import json
import numpy as np
import open3d as o3d
import glob
import rasterio
import subprocess
import random
import shutil


def draw_single_cloud(cloud_path_list):
    if type(cloud_path_list) != list:
        cloud_path_list = [cloud_path_list]
    for idx, cloud_path in enumerate(cloud_path_list):
        pcd = o3d.io.read_point_cloud(cloud_path)
        print(f"Draw {cloud_path}: ", pcd)
        o3d.visualization.draw_geometries(geometry_list=[pcd], window_name=f"Cloud {idx}")


def draw_s2p_cloud(cloud_path_list):
    if type(cloud_path_list) != list:
        cloud_path_list = [cloud_path_list]
    cloud_list = []
    for cloud_path in cloud_path_list:
        pcd = o3d.io.read_point_cloud(cloud_path)
        cloud_list.append(pcd)
        print("Draw: ", pcd)

    o3d.visualization.draw_geometries(cloud_list)


def normal_evaluate(pcd):
    """
    估计点云的法向量
    :param pcd: open3d opend file
    :return:
    """
    radius = 0.01  # 搜索半径
    max_nn = 30  # 最大邻域点数
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    print("normal_evaluate: ", pcd)


def remove_duplicated_point(pcd):
    """
    去除点云中的重复点
    :param pcd: open3d opend file
    :return:
    """
    pcd.remove_duplicated_points()
    print("remove_duplicated_point: ", pcd)


def remove_outlier_point(pcd):
    """
    去除点云中的离群点
    :param pcd: open3d opend file
    :return:
    """
    num_points = 20  # 邻域球内的最少点数，低于该值的点为噪声点
    radius = 0.05  # 邻域半径
    pcd.remove_radius_outlier(nb_points=num_points, radius=radius)
    print("remove_outlier_point: ", pcd)


def merge_s2p_cloud(cloud_path_list, output_file):
    """将s2p每个图像对生成的点云进行合并"""
    if type(cloud_path_list) != list:
        cloud_path_list = [cloud_path_list]

    # 创建一个新的点云对象
    merged_pointcloud = o3d.geometry.PointCloud()
    for cloud_path in cloud_path_list:
        pcd = o3d.io.read_point_cloud(cloud_path)
        merged_pointcloud += pcd  # 将点云添加到合并的点云对象中

    print("Raw cloud: ", merged_pointcloud)
    # 去除重复点
    remove_duplicated_point(merged_pointcloud)
    # 去除噪声点
    remove_outlier_point(merged_pointcloud)
    # 法线估计
    normal_evaluate(merged_pointcloud)

    # 将合并的点云对象写入文件
    o3d.io.write_point_cloud(output_file, merged_pointcloud)


def geojson_polygon(coords_array):
    """
    define a geojson polygon from a Nx2 numpy array with N 2d coordinates delimiting a boundary
    从Nx2 numpy数组定义geojson多边形，其中N个2d坐标界定边界
    """
    from shapely.geometry import Polygon

    # first attempt to construct the polygon, assuming the input coords_array are ordered
    # the centroid is computed using shapely.geometry.Polygon.centroid
    # taking the mean is easier but does not handle different densities of points in the edges
    pp = coords_array.tolist()
    poly = Polygon(pp)
    x_c, y_c = np.array(poly.centroid.xy).ravel()

    # check that the polygon is valid, i.e. that non of its segments intersect
    # if the polygon is not valid, then coords_array was not ordered and we have to do it
    # a possible fix is to sort points by polar angle using the centroid (anti-clockwise order)
    if not poly.is_valid:
        pp.sort(key=lambda p: np.arctan2(p[0] - x_c, p[1] - y_c))

    # construct the geojson
    geojson_polygon = {"coordinates": [pp], "type": "Polygon"}
    geojson_polygon["center"] = [x_c, y_c]
    return geojson_polygon


def lonlat_from_utm(easts, norths, zonestring):
    """
    convert utm to lon-lat
    """
    import pyproj
    proj_src = pyproj.Proj("+proj=utm +zone=%s" % zonestring)
    proj_dst = pyproj.Proj("+proj=latlong")
    return pyproj.transform(proj_src, proj_dst, easts, norths)


def read_DFC2019_lonlat_aoi(aoi_id, dfc_dir):
    if aoi_id[:3] == "JAX":
        zonestring = "17R"
    else:
        raise ValueError("AOI not valid. Expected JAX_(3digits) but received {}".format(aoi_id))
    roi = np.loadtxt(os.path.join(dfc_dir, "Track3-Truth/" + aoi_id + "_DSM.txt"))
    xoff, yoff, xsize, ysize, resolution = roi[0], roi[1], int(roi[2]), int(roi[2]), roi[
        3]  # DSM中数据的含义：前两个值表示左上角起始偏移roi[0] roi[1]，第三个值表示roi区域的长宽roi[2]，最后一个值表示分辨率缩放roi[3]
    ulx, uly, lrx, lry = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff
    xmin, xmax, ymin, ymax = ulx, lrx, uly, lry
    easts = [xmin, xmin, xmax, xmax, xmin]  # 东部区域
    norths = [ymin, ymax, ymax, ymin, ymin]  # 北部区域
    lons, lats = lonlat_from_utm(easts, norths, zonestring)
    lonlat_bbx = geojson_polygon(np.vstack((lons, lats)).T)
    return lonlat_bbx


def load_heuristic_pairs(root_dir, img_dir, heuristic_pairs_file, n_pairs=1):
    # link msi ids to rgb geotiff ids
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    msi_id_to_rgb_id = {}
    for p in img_paths:
        rgb_id = os.path.splitext(os.path.basename(p))[0]
        with rasterio.open(p, "r") as f:
            msi_id = f.tags()["NITF_IID2"].split("-")[0]
            msi_id_to_rgb_id[msi_id] = rgb_id

    selected_pairs_json_paths = []
    with open(heuristic_pairs_file, 'r') as f:
        lines = f.read().split("\n")
    n_selected = 0
    for l in lines:
        tmp = l.split(" ")
        msi_id_l, msi_id_r = os.path.basename(tmp[0]).split("-")[0], os.path.basename(tmp[1]).split("-")[0]
        if msi_id_l in msi_id_to_rgb_id.keys() and msi_id_r in msi_id_to_rgb_id.keys():  # 找到左右图像对
            json_path_l = os.path.join(root_dir, "{}.json".format(msi_id_to_rgb_id[msi_id_l]))
            json_path_r = os.path.join(root_dir, "{}.json".format(msi_id_to_rgb_id[msi_id_r]))
            selected_pairs_json_paths.append((json_path_l, json_path_r))
            n_selected += 1
        if n_selected >= n_pairs:
            break

    return selected_pairs_json_paths


def select_pairs(root_dir, n_pairs=1):

    # load paths of the json files in the training split
    #with open(os.path.join(root_dir, "val.txt"), "r") as f:
    #    json_files = f.read().split("\n")
    #json_paths = [os.path.join(root_dir, bn) for bn in json_files]
    json_paths = glob.glob(os.path.join(root_dir, "*.json"))
    n_train = len(json_paths)

    # list all possible pairs of training samples
    remaining_pairs = []
    n_possible_pairs = 0
    for i in np.arange(n_train):
        for j in np.arange(i + 1, n_train):
            remaining_pairs.append((i, j))
            n_possible_pairs += 1

    # select a random pairs
    selected_pairs_idx, selected_pairs_json_paths = [], []
    for idx in range(n_pairs):
        selected_pairs_idx.append(random.choice(remaining_pairs))
        i, j = selected_pairs_idx[-1][0], selected_pairs_idx[-1][1]
        selected_pairs_json_paths.append((json_paths[i], json_paths[j]))
        remaining_pairs = list(set(remaining_pairs) - set(selected_pairs_idx))

    return selected_pairs_json_paths, n_possible_pairs


def run_s2p(json_path_l, json_path_r, img_dir, out_dir, resolution, prefix="", aoi=None):
    # load json data from the selected pair
    data = []
    for p in [json_path_l, json_path_r]:
        with open(p) as f:
            data.append(json.load(f))

    # create s2p config
    use_pan = True
    if use_pan:
        aoi_id = data[0]["img"][:7]
        if aoi_id in ["JAX_004", "JAX_068"]:
            pan_dir = "/vsicurl/http://138.231.80.166:2332/grss-2019/track_3/Track3-MSI-1/"
        else:
            pan_dir = "/vsicurl/http://138.231.80.166:2332/grss-2019/track_3/Track3-MSI-3/"
        img_path1 = pan_dir + data[0]["img"].replace("RGB", "PAN")
        img_path2 = pan_dir + data[1]["img"].replace("RGB", "PAN")
    else:
        img_path1 = os.path.join(img_dir, data[0]["img"])
        img_path2 = os.path.join(img_dir, data[1]["img"])
    config = {"images": [{"img": img_path1, "rpc": data[0]["rpc"]},  # 生成s2p配置文件
                         {"img": img_path2, "rpc": data[1]["rpc"]}],
              "out_dir": ".",
              "dsm_resolution": resolution,
              "rectification_method": "sift",
              "matching_algorithm": "mgm_multi"}
    if aoi is None:
        config["roi"] = {"x": 0, "y": 0, "w": data[0]["width"], "h": data[0]["height"]}
    else:
        config["roi_geojson"] = aoi

    # sanity check
    if not use_pan:
        for i in [0, 1]:
            if not os.path.exists(config["images"][i]["img"]):
                raise FileNotFoundError("Could not find {}".format(config["images"][i]["img"]))

    # write s2p config to disk
    img_id_l = os.path.splitext(os.path.basename(json_path_l))[0]
    img_id_r = os.path.splitext(os.path.basename(json_path_r))[0]
    s2p_out_dir = os.path.join(out_dir, "{}{}_{}".format(prefix, img_id_l, img_id_r))
    os.makedirs(s2p_out_dir, exist_ok=True)
    config_path = os.path.join(s2p_out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # run s2p and redirect output to log file
    log_file = os.path.join(s2p_out_dir, 'log.txt')
    if not os.path.exists(os.path.join(s2p_out_dir, 'dsm.tif')):
        with open(log_file, 'w') as outfile:
            subprocess.run(['s2p', config_path], stdout=outfile, stderr=outfile)


def generate_cloud_from_s2p(aoi_id, root_dir, dfc_dir, output_dir=".", n_pairs=1, resolution=0.5, crops=False,
                            draw_cloud=False):
    out_dir = os.path.join(output_dir, "s2p_dsms", aoi_id)
    print("Output dir:", out_dir)
    os.makedirs(output_dir, exist_ok=True)

    if crops:
        print("using crops")
        img_dir = os.path.join(dfc_dir, "Track3-RGB-crops/{}".format(aoi_id))
        out_dir += "_crops"
    else:
        img_dir = os.path.join(dfc_dir, "Track3-RGB/{}".format(aoi_id))

    heuristic = True
    heuristic_pairs_file = os.path.join(dfc_dir, "DFC2019_JAX_heuristic_pairs.txt")
    if heuristic and os.path.exists(heuristic_pairs_file):
        selected_pairs_json_paths = load_heuristic_pairs(root_dir, img_dir, heuristic_pairs_file,
                                                         n_pairs=n_pairs)  # 找到指定数量的图像对
        print("{} heuristic pairs selected".format(n_pairs))  # 选择10个启发式配对
    else:
        selected_pairs_json_paths, n_possible_pairs = select_pairs(root_dir, n_pairs=n_pairs)
        print("{} random pairs selected from {} possible".format(n_pairs, n_possible_pairs))

    with open(os.path.join(output_dir, "images_pairs.txt"), "w") as f:
        for pairs in selected_pairs_json_paths:
            f.write(f"{os.path.basename(pairs[0]).replace('json', 'jpg')} "
                    f"{os.path.basename(pairs[1]).replace('json', 'jpg')}\n")

    lonlat_aoi = read_DFC2019_lonlat_aoi(aoi_id, dfc_dir)

    for t, (json_path_l, json_path_r) in enumerate(selected_pairs_json_paths):
        # if len(glob.glob(os.path.join(out_dir, "*/*/*/*/cloud.ply"))) > 0:  # 如果已经存在处理好的点云文件，则不再调用s2p进行处理，此代码用于调试
        #     break
        print("Running s2p ! Pair {} of {}...".format(t + 1, n_pairs))
        # TODO: 调用s2p
        # run_s2p(json_path_l, json_path_r, img_dir, out_dir, resolution, aoi=lonlat_aoi, prefix="{:02}_".format(t))
        print("...done")
    shutil.rmtree("s2p_tmp")
    s2p_ply_paths = glob.glob(os.path.join(out_dir, "*/*/*/*/cloud.ply"))  # 获取所有图像对生成的点云数据
    print("{} s2p ply files found".format(len(s2p_ply_paths)))

    if draw_cloud:
        draw_s2p_cloud(s2p_ply_paths)

    out_dir = os.path.join(out_dir, "Point3D.ply")
    merge_s2p_cloud(s2p_ply_paths, out_dir)  # 合并所有的点云
    # draw_s2p_cloud([out_dir])


if __name__ == '__main__':
    aoi_id = "JAX_068"
    root_dir = "/home/kpl/software/Pycharm/Projects/Geo3DGS/datasets/DFC2019/root_dir/crops_rpcs_ba_v2/JAX_068"
    dfc_dir = "/home/kpl/software/Pycharm/Projects/Geo3DGS/datasets/DFC2019/DFC2019"
    output_dir = "/home/kpl/software/Pycharm/Projects/Geo3DGS/datasets/s2p_result"
    n_pairs = 10
    crops = True
    generate_cloud_from_s2p(aoi_id, root_dir, dfc_dir, output_dir, n_pairs, crops=crops, draw_cloud=False)

    # out_dir = os.path.join(output_dir, "s2p_dsms", aoi_id)
    # s2p_ply_paths = glob.glob(os.path.join(out_dir, "*/*/*/*/cloud.ply"))
    # draw_single_cloud("/home/kpl/software/Pycharm/Projects/Geo3DGS/datasets/s2p_result/s2p_dsms/JAX_068_crops/Point3D.ply")