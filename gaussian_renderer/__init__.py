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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    这段代码定义了一个名为 render 的函数，用于渲染场景。
    函数接受多个参数，包括视点相机 viewpoint_camera、高斯模型 pc、管道 pipe、
        背景颜色 bg_color、缩放修正因子 scaling_modifier 和覆盖颜色 override_color。
    在函数内部，首先创建一个与 pc.get_xyz 相同大小的零张量 screenspace_points，并将其设置为需要计算梯度的张量。
        然后，设置光栅化配置，包括图像的高度、宽度、视角的切线值、背景颜色等。
    接下来，根据提供的参数，确定是否使用预先计算的三维协方差矩阵、预先计算的颜色以及是否进行预计算。
        如果是，则获取预先计算的协方差矩阵、缩放和旋转，或者计算预先计算的颜色。
    然后，使用光栅化器将可见的高斯模型栅格化成图像，并获取它们在屏幕上的半径。
        返回渲染的图像、屏幕空间的点、可见性过滤器以及半径。
    最后，返回渲染结果的字典。
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建零张量。我们将使用它使pytorch返回2D(屏幕空间)的梯度
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0  # [Point_num, 3] 对所有的初始点云坐标初始化
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration 设置光栅化配置
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # 保存高斯渲染参数
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),  # 545
        image_width=int(viewpoint_camera.image_width),  # 980
        tanfovx=tanfovx,  # 0.8446965112441062
        tanfovy=tanfovy,  # 0.4679476755039769
        bg=bg_color,  # [3,] tensor([0., 0., 0.], device='cuda:0')
        scale_modifier=scaling_modifier,  # 1.0
        viewmatrix=viewpoint_camera.world_view_transform,  # [4, 4]
        projmatrix=viewpoint_camera.full_proj_transform,  # [4, 4]
        sh_degree=pc.active_sh_degree,  # 0
        campos=viewpoint_camera.camera_center,  # [3,] tensor([3.1687, 0.1043, 0.9233], device='cuda:0')
        prefiltered=False,  # False
        debug=pipe.debug  # False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)  # 初始化高斯渲染模型参数

    means3D = pc.get_xyz  # [Point_num, 3]
    means2D = screenspace_points  # [Point_num, 3]
    opacity = pc.get_opacity  # [Point_num, 1] 透明度

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 如果提供了预计算的3d协方差，请使用它。如果没有，那么它将由光栅缩放/旋转计算。
    scales = None  # [Point_num, 3]
    rotations = None  # [Point_num, 4]
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)  # 计算协方差矩阵
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 如果提供了预先计算的颜色，就使用它们。否则，如果希望在Python中从SHs中预计算颜色，请执行。如果没有，则SH->RGB转换将由光栅完成。
    """
    这段代码片段描述了如何在Python中将点云数据转换为RGB颜色。它提供了两种选项：
        如果提供预计算的颜色，则使用它们。
        否则，如果想在Python中从SHs中预计算颜色，则执行此操作。
        以下是代码片段的详细说明：
        
        如果提供预计算的颜色，则直接使用它们。
        否则，如果想在Python中从SHs中预计算颜色，则执行以下操作：
        首先，从点云数据中提取必要的特征和属性，例如法向量和点云位置。
        然后，将法向量归一化，以便将其用作SHs的输入。
        使用eval_sh函数计算SHs到RGB的转换。eval_sh函数的参数包括：
        PC（点云数据）：包含点云特征和属性的对象。
        active_sh_degree（球面谐波阶数）：SHs的阶数。
        shs_view（SHs视图）：将SHs从3D转换为2D，以便可以将其传递给eval_sh函数。
        dir_pp_normalized（方向归一化）：归一化的点云方向。
        最后，将计算出的RGB值（sh2rgb）添加到0.5以使其在[0, 1]范围内，并将结果作为预计算的颜色存储。
    """
    shs = None  # [Point_num, 16, 3]
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # 使用预先计算好的颜色
        else:
            shs = pc.get_features  # 由光栅器计算颜色
    else:
        colors_precomp = override_color

    # TODO: 需要添加train时不渲染depth，render渲染深度的代码
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 将可见的高斯分布栅格化成图像，获取其半径(在屏幕上)。
    rendered_image, radii = rasterizer(  # 原始的3DGS，不进行深度渲染
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # rendered_image, radii, depth = rasterizer(  # 经过修改的diff-gaussian，只进行forwards才输出深度，depth不能进行backwards，因此depth只有在进行render是才能启用
    #     means3D=means3D,
    #     means2D=means2D,
    #     shs=shs,
    #     colors_precomp=colors_precomp,
    #     opacities=opacity,
    #     scales=scales,
    #     rotations=rotations,
    #     cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 那些被截锥体剔除或半径为0的高斯分布是不可见的。在拆分条件中，它们将被排除在值更新之外。
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,  # 能见度过滤器
            "radii": radii}  # 半径

    # 渲染深度
    # return {"render": rendered_image,
    #         "viewspace_points": screenspace_points,
    #         "visibility_filter": radii > 0,  # 能见度过滤器
    #         "radii": radii,
    #         "depth": depth}  # 半径
