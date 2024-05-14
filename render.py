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
from torchvision import transforms
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
# from arguments import ModelParams, PipelineParams, get_combined_args
from arguments.parameters import ModelParams, PipelineParams, get_combined_args, extract
from gaussian_renderer import GaussianModel


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depths")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        # depth_image = rendering['depth']
        # depth_image = transforms.ToPILImage()(depth_image)
        # depth_image.save(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                       background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                       background)


def render_main():
    """重写渲染主函数
    将原先隐式的参数进行显式化重写，方便阅读和调参
    代码主体与原先仍保持一致
    """
    parser = ArgumentParser(description="Testing script parameters")
    # 加载两个模块的参数
    model = ModelParams(parser).parse_args()
    pipeline, before_extract_pipeline = extract(model, PipelineParams(parser).parse_args())

    # render.py脚本显式参数
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)     # 加载保存的模型中的配置信息
    args.source_path = os.path.abspath(args.source_path)
    args.model_path = os.path.abspath(args.model_path)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model, args.iteration, pipeline, args.skip_train, args.skip_test)


if __name__ == "__main__":
    # # Set up command line argument parser
    # parser = ArgumentParser(description="Testing script parameters")
    # model = ModelParams(parser, sentinel=True)
    # pipeline = PipelineParams(parser)
    # parser.add_argument("--iteration", default=-1, type=int)
    # parser.add_argument("--skip_train", action="store_true")
    # parser.add_argument("--skip_test", action="store_true")
    # parser.add_argument("--quiet", action="store_true")
    # args = get_combined_args(parser)
    # print("Rendering " + args.model_path)
    #
    # # Initialize system state (RNG)
    # safe_state(args.quiet)
    #
    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

    render_main()