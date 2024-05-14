# -*- coding: utf-8 -*- 
# @Author: Kang Peilun
# @Email: 374774222@qq.com
# @File: rpc_utils.py
# @Project: Geo3DGS
# @Time: 2023/12/22 上午10:39
# @Des:

import rpcm
from rpcm import rpc_model
import os
import os.path as op
from glob import glob
from tqdm import tqdm


def load_image_path(image_path, format):
    """load image path
    :image_path: image path
    :format: image format
    """
    image_path_list = glob(op.join(image_path, '*.{}'.format(format)))
    return image_path_list


def check_dirs(path):
    if not op.exists(path):
        os.makedirs(path)
    return path


def read_rpc_from_geotiff(image_path_list):
    for image_path in tqdm(image_path_list, total=len(image_path_list), desc="Reading GeoTiff"):
        rpc = rpc_model.rpc_from_geotiff(image_path)
        print(rpc)


if __name__ == '__main__':
    read_rpc_from_geotiff(load_image_path("/home/kpl/software/Pycharm/Projects/Geo3DGS/datasets/DFC_3DGS/input_tif", "tif"))