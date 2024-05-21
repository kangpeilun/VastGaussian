# Author: Peilun Kang
# Contact: kangpeilun@nefu.edu.cn
# License: Apache Licence
# Project: VastGaussian
# File: decouple_appearance_model.py
# Time: 5/15/24 2:30 PM
# Des: Decoupled Appearance Modeling 外观解耦建模

import torch
from torch import nn
from torch import optim


class DecoupleAppearanceModel(nn.Module):
    def __init__(self, downsample=32, upscale_factor=2, embedding=64, block=4):
        super(DecoupleAppearanceModel, self).__init__()
        self.embedding = embedding
        # self.appearance_embedding = nn.Conv2d(3, embedding, kernel_size=3, stride=1, padding=1)  # 使用卷积生成外观embedding
        self.appearance_embedding = nn.Embedding(2_0000, 64)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)  # 使用PixelShuffle实现2倍上采样，不改变通道数
        self.down_sample = nn.Conv2d(3, 3, kernel_size=downsample, stride=downsample)  # 使用卷积网络实现32下采样
        self.conv2d = nn.Conv2d(3 + embedding, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.PixelConvRelu = nn.ModuleList()

        # 构建网络
        for idx in range(block):
            self.PixelConvRelu.append(nn.Sequential(
                self.pixel_shuffle,
                nn.Conv2d(256 // ((upscale_factor**2)*(2**idx)), 256 // (upscale_factor*(2**idx)), kernel_size=3, stride=1, padding=1),
                self.relu,
            ))

        self.ConvReluConv = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            self.relu,
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        )

    def optimize(self, model, lr=0.001):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return optimizer, scheduler

    def forward(self, image):
        """
        外观解耦网络
        :param image: [3, W, H]
        :return:
        """
        rendered_image = image.clone()
        H, W = image.size(1), image.size(2)
        x = self.down_sample(image)  # [3, H/32, W/32] 下采样

        indices = torch.arange(0, x.size(1)*x.size(2), device="cuda").long()  # 生成降采样后像素个数的embedding [h*w,]
        appearance_embedding = self.appearance_embedding(indices).view(self.embedding, x.size(1), x.size(2))  # [h*w, 64] -> [64, h, w]
        x = torch.cat((x, appearance_embedding), dim=0)  # [67, H, W]

        x = self.conv2d(x)  # [256, H/32, W/32]
        for block in self.PixelConvRelu:
            x = block(x)
        # [16, H/2, W/2]
        x = nn.functional.interpolate(x.unsqueeze(0), size=(H, W), mode='bilinear').squeeze()  # [16, H, W]
        transformation_map_temp = self.ConvReluConv(x)  # [3, H, W]
        # print("transformation_map", transformation_map)
        # print("min:", transformation_map.min(), " max:", transformation_map.max())

        transformation_map = torch.sigmoid(transformation_map_temp)  # TODO: 如果不使用sigmoid，可视化的转换图颜色会非常奇怪，不是灰色的，而是一片绿色或者蓝色等, 暂不清楚原因

        decouple_image = rendered_image * transformation_map  # 将渲染图像乘以变换图 得到外观解耦后的图像
        return decouple_image, transformation_map
