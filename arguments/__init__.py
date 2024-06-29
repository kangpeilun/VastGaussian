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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""  # 包含COLMAP或Synthetic NeRF数据集的源目录的路径。
        self._model_path = ""   # 存储训练模型的路径(默认为output/<random>)。
        self._images = "images"  # COLMAP图像(默认为图像)的备选子目录。
        self._resolution = -1  # -1
        # 指定训练前加载图像的分辨率。如果提供1、2、4或8，分别使用原始、1/2、1/4或1/8分辨率。对于所有其他值，将宽度重新缩放到给定的数字，同时保持图像的长宽。
        # 如果未设置且输入图像宽度超过1.6K像素，则输入将自动重新缩放到此目标。
        self._white_background = False
        self.data_device = "cuda"  # 指定源图像数据的位置，默认为cuda，如果在大型/高分辨率数据集上训练，建议使用cpu，将减少VRAM消耗，但会稍微减慢训练速度。
        self.eval = False  # 添加此标志以使用mipnerf360风格的培训/测试分割进行评估。
        self.llffhold = 83  # 可以被llffhold整除的图像索引，作为测试机
        # New Params
        self.exp_name = ""
        # Manhattan
        self.manhattan = False  # 是否需要曼哈顿对齐
        self.plantform = "cc"  # cloudcompare: cc, threejs: tj  # 使用哪种平台进行曼哈顿对齐
        self.pos = ""  # 点云平移，平移向量，如果使用threejs，则pos和rot的参数个数均为三个，如果使用cloudcompare则，rot为9个数，pos为3个数
        self.rot = ""  # 点云平移，如果处理平台为cloudcompare，则rot为旋转矩阵，否则用threejs处理rot就为三个旋转向量
        self.man_trans = None  # 指定经过曼哈顿对齐后的点云坐标相对于初始点云坐标的变换矩阵
        # Data Partition
        self.m_region = 3
        self.n_region = 3
        self.extend_rate = 0.2
        self.visible_rate = 0.25

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False  # 标志，使管道用PyTorch计算向前和向后的SHs，而不是我们的。
        self.compute_cov3D_python = False  # 标志，使管道用PyTorch计算3D协方差的正向和反向，而不是我们的。
        self.debug = False  # 如果遇到错误，启用调试模式。如果栅格化失败，则会创建一个转储文件，您可以将其转发给我们，以便我们查看。
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000  # 要训练的总迭代数，默认为30_000。
        self.position_lr_init = 0.00016  # 初始3D位置学习率默认为0.00016。
        self.position_lr_final = 0.0000016  # 最终3D位置学习率，默认为0.0000016。
        self.position_lr_delay_mult = 0.01  # 位置学习率乘数(参见Plenoxels)，默认为0.01。
        self.position_lr_max_steps = 30_000  # 位置学习率从初始到最终的步数(从0开始)。默认为30_000。
        self.feature_lr = 0.0025  # 球面谐波具有学习率，默认为0.0025。
        self.opacity_lr = 0.05    # 不透明学习率默认为0.05。
        self.scaling_lr = 0.005   # 缩放学习率默认为0.005。
        self.rotation_lr = 0.001  # 旋转学习率默认为0.001。
        self.percent_dense = 0.01  # 一个点必须超过场景范围的百分比(0-1)才能强制致密化，默认为0.01。通过百分比来限制多大的高斯应该被split，多小的高斯应该被clone
        self.lambda_dssim = 0.2  # SSIM对总损失的影响从0到1,0.2默认。
        # VastGaussian Settings
        # The densification starts at the 1000th iteration and ends at the
        # 30, 000th iteration, with an interval of 200 iterations.
        self.densification_interval = 200  # 密集化的频率，默认为100(每100次迭代)。
        self.opacity_reset_interval = 3000  # 重置不透明度的频率，默认为3_000。优化可能会遇到靠近输入摄像头的漂浮物,也就是致密化产生不必要的高斯点，因此将将不透明度设置为0
        self.densify_from_iter = 1000  # 开始致密化的迭代，默认为500。
        self.densify_until_iter = 30_000  # 迭代时停止致密化，默认为15_000。
        self.densify_grad_threshold = 0.0002  # 决定点是否应该基于2D位置梯度进行密度化的限制，默认值为0.0002。
        self.random_background = False

        # Appearance Decouple
        self.appearance_embeddings_lr = 0.001  # AE的学习率
        self.appearance_network_lr = 0.001     # 外观解耦网络的学习率
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

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
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
