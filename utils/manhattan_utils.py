# -*- coding: utf-8 -*-
#        Data: 2024-06-21 15:59
#     Project: VastGaussian
#   File Name: manhattan_utils.py
#      Author: KangPeilun
#       Email: 374774222@qq.com 
# Description:

import math
import numpy as np

def create_man_rans(position, rotation):
    # create manhattan transformation matrix for threejs
    # The angle is reversed because the counterclockwise direction is defined as negative in three.js
    rot_x = np.array([[1, 0, 0],
                      [0, math.cos(np.deg2rad(-rotation[0])), -math.sin(np.deg2rad(-rotation[0]))],
                      [0, math.sin(np.deg2rad(-rotation[0])),  math.cos(np.deg2rad(-rotation[0]))]])
    rot_y = np.array([[ math.cos(np.deg2rad(-rotation[1])), 0, math.sin(np.deg2rad(-rotation[1]))],
                      [0, 1, 0],
                      [-math.sin(np.deg2rad(-rotation[1])), 0, math.cos(np.deg2rad(-rotation[1]))]])
    rot_z = np.array([[math.cos(np.deg2rad(-rotation[2])), -math.sin(np.deg2rad(-rotation[2])), 0],
                      [math.sin(np.deg2rad(-rotation[2])),  math.cos(np.deg2rad(-rotation[2])), 0],
                      [0, 0, 1]])

    rot = rot_z @ rot_y @ rot_x
    man_trans = np.zeros((4, 4))
    man_trans[:3, :3] = rot.transpose()
    man_trans[:3, -1] = np.array(position).transpose()
    man_trans[3, 3] = 1

    return man_trans


def get_man_trans(lp):
    lp.pos = [float(pos) for pos in lp.pos.split(" ")]
    lp.rot = [float(rot) for rot in lp.rot.split(" ")]

    man_trans = None
    if lp.manhattan and lp.plantform == "tj":  # threejs
        man_trans = create_man_rans(lp.pos, lp.rot)
        lp.man_trans = man_trans
    elif lp.manhattan and lp.plantform == "cc":  # cloudcompare 如果处理平台为cloudcompare，则rot为旋转矩阵
        rot = np.array(lp.rot).reshape([3, 3])
        man_trans = np.zeros((4, 4))
        man_trans[:3, :3] = rot
        man_trans[:3, -1] = np.array(lp.pos)
        man_trans[3, 3] = 1

    return man_trans