# -*- coding: utf-8 -*-
# date: 2023/10/12
# Project: gaussian-splatting
# File Name: parameters.py
# Description: 重新组织参数形式，使之看起来更直观
# Author: Anefuer_kpl
# Email: 374774222@qq.com

from argparse import ArgumentParser, Namespace
import math
import os
import numpy as np

"""
    在原来的版本中，作者将所有的参数写入三个类中分别进行动态加载
    这种写法可以减少主函数中的参数数量，显得更简洁，但是这种方式不够直观
    因此我将所有的参数摘出来，重写了配置文件，只是修改了参数的呈现方式，
    并没有对代码整体的结构进行修改
"""
def create_man_rans(position, rotation):

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


def ModelParams(parser):
    # 这块是ModelParams，使用默认值即可，必须传入source_path参数
    parser.add_argument("--source_path", "-s", default='', required=True)  # 包含COLMAP或Synthetic NeRF数据集的源目录的路径。
    parser.add_argument("--model_path", "-m", default='')  # 存储训练模型的路径(默认为output/<random>)。
    parser.add_argument("--images", "-i", type=str, default="images")  # COLMAP图像(默认为图像)的备选子目录。
    parser.add_argument("--eval", type=bool, default=False)  # 添加此标志以使用mipnerf360风格的培训/测试分割进行评估。
    # 指定训练前加载图像的分辨率。如果提供1、2、4或8，分别使用原始、1/2、1/4或1/8分辨率。对于所有其他值，将宽度重新缩放到给定的数字，同时保持图像的长宽。
    # 如果未设置且输入图像宽度超过1.6K像素，则输入将自动重新缩放到此目标。
    parser.add_argument("--resolution", "-r", type=int, default=-1)  # -1
    parser.add_argument("--data_device", type=str,
                        default="cuda")  # 指定源图像数据的位置，默认为cuda，如果在大型/高分辨率数据集上训练，建议使用cpu，将减少VRAM消耗，但会稍微减慢训练速度。
    parser.add_argument("--white_background", "-w", type=bool, default=False)  # 添加此标志以使用白色背景而不是黑色(默认)，例如，用于评估NeRF合成数据集。
    parser.add_argument("--sh_degree", type=int, default=3)  # 要使用的球面谐波阶数(不大于3)。默认为3。

    # 新增的可选参数，用于实验调整参数
    parser.add_argument("--exp_name", type=str, default="building_downsample", help="experiment name, if have the same name dir, mkdir a new one like exp_name_1, exp_name_2 ...")  # 为每次实验命名
    # Manual Manhattan alignment
    parser.add_argument("--manhattan", action="store_true")   # 是否需要曼哈顿对齐
    parser.add_argument("--plantform", type=str, default="cloudcompare", choices=["cloudcompare", "threejs"])  # 使用哪种平台进行曼哈顿对齐
    parser.add_argument("--pos", nargs="+", type=float, default=[0, 0, 0])         # 点云平移，平移向量，如果使用threejs，则pos和rot的参数个数均为三个，如果使用cloudcompare则，rot为9个数，pos为3个数
    parser.add_argument("--rot", nargs="+", type=float, default=[0, 0, 0])         # 点云平移，如果处理平台为cloudcompare，则rot为旋转矩阵，否则用threejs处理rot就为三个旋转向量
    parser.add_argument("--man_trans", default=None)  # 指定经过曼哈顿对齐后的点云坐标相对于初始点云坐标的变换矩阵
    # data partition params
    parser.add_argument("--m_region", type=int, default=3, help="the number of regions in the x direction")  # 划分区域的数量，论文作者提醒虽然论文里写的是8块，但实操时用的是9块
    parser.add_argument("--n_region", type=int, default=3, help="the number of regions in the z direction")
    parser.add_argument("--extend_rate", type=float, default=0.7, help="The rate of boundary expansion")
    parser.add_argument("--visible_rate", type=float, default=0.25, help="Airspace-aware visibility rate")

    parser.add_argument("--num_gpus", type=int, default=1, help="if =1 train model on 1 GPU, if =n train model on n GPUs")

    # Pre_train DAM
    parser.add_argument("--pre_train_iteration", type=int, default=15_000, help="while m_region=1 and n_region=1, save {pre_train_iteration} DAM model")

    return parser


def OptimizationParams(parser):
    # 下面是OptimizationParams，使用默认值即可
    parser.add_argument("--iterations", type=int, default=30_000)  # 要训练的总迭代数，默认为30_000。
    parser.add_argument("--feature_lr", type=float, default=0.0025)  # 球面谐波具有学习率，默认为0.0025。
    parser.add_argument("--opacity_lr", type=float, default=0.05)  # 不透明学习率默认为0.05。
    parser.add_argument("--scaling_lr", type=float, default=0.005)  # 缩放学习率默认为0.005。
    parser.add_argument("--rotation_lr", type=float, default=0.001)  # 旋转学习率默认为0.001。
    parser.add_argument("--position_lr_max_steps", type=int, default=30_000)  # 位置学习率从初始到最终的步数(从0开始)。默认为30_000。
    parser.add_argument("--position_lr_init", type=float, default=0.00016)  # 初始3D位置学习率默认为0.00016。
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)  # 最终3D位置学习率，默认为0.0000016。
    parser.add_argument("--position_lr_delay_mult", type=float, default=0.01)  # 位置学习率乘数(参见Plenoxels)，默认为0.01。
    parser.add_argument("--densify_from_iter", type=int, default=500)  # 开始致密化的迭代，默认为500。
    parser.add_argument("--densify_until_iter", type=int, default=15_000)  # 迭代时停止致密化，默认为15_000。
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002)  # 决定点是否应该基于2D位置梯度进行密度化的限制，默认值为0.0002。
    parser.add_argument("--densification_interval", type=float, default=100)  # 密集化的频率，默认为100(每100次迭代)。
    parser.add_argument("--opacity_reset_interval", type=int, default=3000)  # 重置不透明度的频率，默认为3_000。优化可能会遇到靠近输入摄像头的漂浮物,也就是致密化产生不必要的高斯点，因此将将不透明度设置为0
    parser.add_argument("--lambda_dssim", type=float, default=0.2)  # SSIM对总损失的影响从0到1,0.2默认。
    parser.add_argument("--percent_dense", type=float, default=0.01)  # 一个点必须超过场景范围的百分比(0-1)才能强制致密化，默认为0.01。通过百分比来限制多大的高斯应该被split，多小的高斯应该被clone

    # Appearance Decouple
    parser.add_argument("--appearance_embeddings_lr", type=float, default=0.001)  # AE的学习率
    parser.add_argument("--appearance_network_lr", type=float, default=0.001)  # 外观解耦网络的学习率

    return parser


def PipelineParams(parser):
    # 这块是PipelineParams，使用默认值即可
    parser.add_argument("--convert_SHs_python", type=bool, default=False)  # 标志，使管道用PyTorch计算向前和向后的SHs，而不是我们的。
    parser.add_argument("--compute_cov3D_python", type=bool, default=False)  # 标志，使管道用PyTorch计算3D协方差的正向和反向，而不是我们的。
    parser.add_argument("--debug", type=bool, default=False)  # 如果遇到错误，启用调试模式。如果栅格化失败，则会创建一个转储文件，您可以将其转发给我们，以便我们查看。

    return parser


def extract(args1, args2):
    """将args1的参数从args2中剔除
    :return 返回剔除参数后的args2, 以及没有剔除前的args2
    """
    args1_dict = vars(args1)
    args2_dict = vars(args2)
    before_extract_args2 = vars(args2).copy()  # 复制一份没有剔除前的args2

    for key1, value1 in args1_dict.items():
        if key1 in args2_dict:
            args2_dict.pop(key1)

    return Namespace(**args2_dict), Namespace(**before_extract_args2)


def get_combined_args(parser: ArgumentParser):
    """从训练好的模型目录中加载训练时保存的配置信息"""
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args()

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v

    return Namespace(**merged_dict)
