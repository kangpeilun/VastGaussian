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
import copy
import glob
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
# from scene import Scene, GaussianModel
from VastGaussian_scene.datasets import BigScene, PartitionScene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
# from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments.parameters import ModelParams, PipelineParams, OptimizationParams, extract, create_man_rans

from VastGaussian_scene.seamless_merging import seamless_merge
from utils.general_utils import PILtoTorch


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

WARNED = False

# https://github.com/autonomousvision/gaussian-opacity-fields
def decouple_appearance(image, gaussians, view_idx):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    H, W = image.size(1), image.size(2)
    # down sample the image
    # print("H", H, "W", W)
    # print("image", image, image.size())
    crop_image_down = torch.nn.functional.interpolate(image[None], size=(H // 32, W // 32), mode="bilinear", align_corners=True)[0]

    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H // 32, W // 32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down, H, W).squeeze()
    transformed_image = mapping_image * image

    return transformed_image, mapping_image


def load_image_while_training(args, image_path):
    """在训练时，每加载一次相机，同时载入对应的图片
    """
    image = Image.open(image_path)
    orig_w, orig_h = image.width, image.height
    # print(image_path, orig_w, orig_h)
    resolution_scale = 1.0
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))  # 计算下采样后，图片的尺寸
    else:  # should be a type that converts to float 应该是转换为float的类型吗
        if args.resolution == -1:  # 即使没有设置下采样的倍率，也会自动判断图片的宽度是否大于1600，如果大于，则自动进行下采样，并计算下采样的倍率
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(image, resolution)  # [C, H, W]

    original_image = resized_image_rgb[:3, ...]
    height = original_image.size(1)
    width = original_image.size(2)

    return original_image, height, width


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    tb_writer = prepare_output_and_logger(dataset)
    big_scene = BigScene(dataset)  # 这段代码整个都是加载数据集，同时包含高斯模型参数的加载

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=dataset.data_device)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    for partition_id in range(len(big_scene.partition_data)):
        gaussians = GaussianModel(dataset)
        partition_scene = PartitionScene(dataset, gaussians, partition_id, big_scene.partition_data[partition_id])
        gaussians.training_setup(opt)

        viewpoint_stack = None
        ema_loss_for_log = 0.0

        first_iter = 0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc=f"Training progress partition:{partition_id}")
        first_iter += 1
        # 执行训练循环
        for iteration in range(first_iter, opt.iterations + 1):
            iter_start.record()

            gaussians.update_learning_rate(iteration)  # 根据迭代次数，更新优化器学习率

            # Every 1000 its we increase the levels of SH up to a maximum degree
            # 每1000次，我们将SH水平提高到最大程度
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera 随机选择一个相机
            if not viewpoint_stack:
                viewpoint_stack = partition_scene.getTrainCameras().copy()
            rand_image_id = randint(0, len(viewpoint_stack) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_image_id)  # 从相机列表中随机选择一个相机

            original_image, new_height, new_width = load_image_while_training(dataset, os.path.join(dataset.source_path, "images", viewpoint_cam.image_name+".jpg"))
            # print("image_path: ", os.path.join(dataset.source_path, "images", viewpoint_cam.image_name+".jpg"), new_height, new_width)
            viewpoint_cam.image_width = new_width
            viewpoint_cam.image_height = new_height
            viewpoint_cam.original_image = original_image

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            # depth参数不参与梯度的更新和模型训练，只有在只进行render时渲染出depth图
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            # 外观解耦模型
            decouple_image, transformation_map = decouple_appearance(image, gaussians, viewpoint_cam.uid)
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()  # 获取ground truth图像
            # Ll1 = l1_loss(image, gt_image)
            Ll1 = l1_loss(decouple_image, gt_image)  # 使用外观解耦后的图像与gt计算损失
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                        1.0 - ssim(image, gt_image))  # loss = L1_loss + SSIM_loss(图像质量损失)  lambda_dssim控制ssim对总损失的影响，默认为0.2
            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(dataset, tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                testing_iterations, partition_scene, render, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    partition_scene.save(iteration)

                # Densification  致密化
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning  跟踪图像空间中的最大半径以进行修剪
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:  # 当迭代次数大于500,并且迭代次数能够整除100时，每隔100个迭代对点云进行优化
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, partition_scene.cameras_extent, size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)


                # 每500轮保存一次中间外观解耦图像
                if iteration % 10 == 0:
                    decouple_image = decouple_image.cpu()
                    decouple_image = transforms.ToPILImage()(decouple_image)
                    save_dir = os.path.join(partition_scene.model_path, "decouple_images")
                    if not os.path.exists(save_dir): os.makedirs(save_dir)
                    decouple_image.save(f"{save_dir}/decouple_image_{partition_id}_{viewpoint_cam.uid}_{iteration}.png")

                    transformation_map = transformation_map.cpu()
                    transformation_map = transforms.ToPILImage()(transformation_map)
                    transformation_map.save(f"{save_dir}/transformation_map_{partition_id}_{viewpoint_cam.uid}_{iteration}.png")

                    image = image.cpu()
                    image = transforms.ToPILImage()(image)
                    image.save(f"{save_dir}/render_image_{partition_id}_{viewpoint_cam.uid}_{iteration}.png")

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), partition_scene.model_path + "/chkpnt" + str(iteration) + ".pth")


    # seamless_merging 无缝合并
    print("Merging Partitions...")
    all_point_cloud_dir = glob.glob(os.path.join(dataset.model_path, "point_cloud", "*"))

    for point_cloud_dir in all_point_cloud_dir:
        seamless_merge(dataset.model_path, point_cloud_dir)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    model_path = os.path.join("./output/", args.exp_name)
    # 如果这个文件存在，就在这个文件名的基础上创建新的文件夹，文件名后面跟上1,2,3
    if os.path.exists(model_path):
        base_name = os.path.basename(model_path)
        dir_name = os.path.dirname(model_path)
        file_name, file_ext = os.path.splitext(base_name)
        counter = 1
        while os.path.exists(os.path.join(dir_name, f"{file_name}_{counter}{file_ext}")):
            counter += 1
        new_folder_name = f"{file_name}_{counter}{file_ext}"
        model_path = os.path.join(dir_name, new_folder_name)
    args.model_path = model_path
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        var_dict = copy.deepcopy(vars(args))
        del_var_list = ["manhattan", "man_trans", "pos", "rot",
                        "m_region", "n_region", "extend_rate", "visible_rate"]  # 删除多余的变量，防止无法使用SIBR可视化
        for del_var in del_var_list:
            del var_dict[del_var]
        cfg_log_f.write(str(Namespace(**var_dict)))


    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(dataset, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: PartitionScene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = [
            # {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train',
             'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                         range(5, 30, 5)]}]

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    original_image, new_height, new_width = load_image_while_training(dataset,
                                                                                      os.path.join(dataset.source_path,
                                                                                                   "images",
                                                                                                   viewpoint.image_name + ".jpg"))
                    viewpoint.image_width = new_width
                    viewpoint.image_height = new_height
                    viewpoint.original_image = original_image

                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def train_main():
    """重写训练主函数
    将原先隐式的参数进行显式化重写，方便阅读和调参
    代码主体与原先仍保持一致
    参数详情见：arguments/parameters.py
    """
    parser = ArgumentParser(description="Training Script Parameters")
    # 三个模块里的参数
    lp = ModelParams(parser).parse_args()
    op, before_extract_op = extract(lp, OptimizationParams(parser).parse_args())
    pp, before_extract_pp = extract(before_extract_op, PipelineParams(parser).parse_args())

    if lp.manhattan and lp.plantform == "threejs":
        man_trans = create_man_rans(lp.pos, lp.rot)
        lp.man_trans = man_trans

    # 使用cloudcompare处理后，将经过多次translate/rotate的得到的多个变换矩阵依次左乘即可得到最终的变换矩阵
    # A1 = np.array([0.825355350971, 0.036096323282, 0.563458621502, -1.877131938934,
    #                0.028794845566, 0.993964672089, -0.105854183435, 0.351995319128,
    #                -0.563878893852, 0.103592015803, 0.819334685802, 0.501181662083,
    #                0, 0, 0, 1]).reshape([4, 4])
    # A2 = np.array([0.997768461704, 0.005278629251, -0.066559724510, -5.802182197571,
    #                0.005290360656, 0.999985992908, 0.000000000000, 4.109899044037,
    #                0.066558793187, 0.000352124945, 0.997782468796, 0.139409780502,
    #                0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]).reshape([4, 4])
    # man_trans = A2 @ A1
    # array([[0.86119716, 0.03436749, 0.50710779, -7.70662571], rot 0.86119716 0.03436749 0.50710779 0.03316087 0.99414171 -0.1028718 -0.50768368 0.10611482 0.85498364
    #        [0.03316087, 0.99414171, -0.1028718, 4.45195873],  pos -7.70662571 4.45195873 0.51466437
    #        [-0.50768368, 0.10611482, 0.85498364, 0.51466437],
    #        [0., 0., 0., 1.]])

    elif lp.manhattan and lp.plantform == "cloudcompare":  # 如果处理平台为cloudcompare，则rot为旋转矩阵
        rot = np.array(lp.rot).reshape([3, 3])
        man_trans = np.zeros((4, 4))
        man_trans[:3, :3] = rot
        man_trans[:3, -1] = np.array(lp.pos)
        man_trans[3, 3] = 1
        lp.man_trans = man_trans

    # train.py脚本显式参数
    parser.add_argument("--ip", type=str, default='127.0.0.1')  # 启动GUI服务器的IP地址，默认为127.0.0.1。
    parser.add_argument("--port", type=int, default=6009)  # 用于GUI服务器的端口，默认为6009。
    parser.add_argument("--debug_from", type=int, default=-1)  # 调试缓慢。您可以指定一个迭代(从0开始)，之后上述调试变为活动状态。
    parser.add_argument("--detect_anomaly", default=False)  #
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[10, 100, 1000, 7_000, 30_000])  # 训练脚本在测试集上计算L1和PSNR的间隔迭代，默认为7000 30000。
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[100, 7_000, 30_000, 60_000])  # 训练脚本保存高斯模型的空格分隔迭代，默认为7000 30000 <迭代>。
    parser.add_argument("--quiet", default=False)  # 标记以省略写入标准输出管道的任何文本。
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])  # 空格分隔的迭代，在其中存储稍后继续的检查点，保存在模型目录中。
    parser.add_argument("--start_checkpoint", type=str, default=None)  # 路径保存检查点继续训练。
    args = parser.parse_args()
    args.save_iterations.append(args.iterations)
    args.source_path = os.path.abspath(args.source_path)  # 将相对路径转换为绝对路径

    if args.manhattan:
        print("Need to perform Manhattan World Hypothesis based alignment")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp, op, pp, args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    train_main()


