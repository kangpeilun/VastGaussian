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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", params=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.params = params

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        """
        image.clamp(low, high)
        将图像的所有波段中的值都夹在指定范围内。
        低于该范围低值的值被设置为低值，高于该范围的高值的值被设置为高值
        
        使用 clamp 方法将图像 image 的值夹在 0.0 到 1.0 的范围内，并将结果转移到 self.data_device 上，并将结果赋值给 self.original_image。
            同时，获取图像的宽度和高度，并分别赋值给 self.image_width 和 self.image_height。
        如果 gt_alpha_mask 不为空，则将 self.original_image 乘以 gt_alpha_mask，以将图像中的非透明部分保留下来。
            否则，将 self.original_image 乘以一个全为 1 的张量，以保留整个图像。
        接下来，设置相机的远近裁剪平面，将 self.zfar 设置为 100.0，将 self.znear 设置为 0.01。
        然后，将平移向量 trans 和缩放因子 scale 传入 getWorld2View2 函数，并将结果转置后转移到 cuda 设备上，并将结果赋值给 self.world_view_transform。
        接着，调用 getProjectionMatrix 函数，传入远近裁剪平面的值以及水平和垂直方向的视场角，将结果转置后转移到 cuda 设备上，并将结果赋值给 self.projection_matrix。
        然后，通过矩阵乘法计算出 self.full_proj_transform，表示世界坐标系到投影坐标系的变换矩阵。
        最后，通过求 self.world_view_transform 的逆矩阵，并提取其中的前三行前三列，得到相机的中心点坐标，并将其赋值给 self.camera_center。
        """
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)  # 2G显存 执行这行代码时会报显存不足

        # 距离相机平面znear和zfar之间且在视锥内的物体才会被渲染
        self.zfar = 100.00  # 最远能看到多远
        self.znear = 0.01   # 最近能看到多近

        self.trans = trans  # 相机中心的平移
        self.scale = scale  # 相机中心坐标的缩放
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()  # W2C世界到相机坐标系的变换矩阵，4×4  注意这里使用transpose(0, 1)进行了一下转置
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()  # 裁减空间->NDC空间，用于将坐标压缩到0-1之间
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)  # 世界->相机->裁减空间->NDC空间
        self.camera_center = self.world_view_transform.inverse()[3, :3]  # 相机中心在世界坐标系下的坐标 world_view_transform进行了一下转置，所以取[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

