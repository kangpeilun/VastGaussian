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
import glob
import os
import torch
from PIL import Image
from torchvision import transforms
from random import randint
import torch.multiprocessing as mp

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
from arguments.parameters import ModelParams, PipelineParams, OptimizationParams, extract

from VastGaussian_scene.decouple_appearance_model import DecoupleAppearanceModel
from VastGaussian_scene.seamless_merging import seamless_merge

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

mp.set_start_method('spawn', force=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train_partition(opt, first_iter, iter_start, iter_end,
                    progress_bar, tb_writer, testing_iterations, saving_iterations,
                    partition_scene, debug_from, pipe, dataset, background,
                    gaussians, DAModel, partition_id, checkpoint_iterations, synchronizer):
    optimizer, scheduler = DAModel.optimize()

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
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))  # 从相机列表中随机选择一个相机

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        # depth参数不参与梯度的更新和模型训练，只有在只进行render时渲染出depth图
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
            "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # 外观解耦模型
        decouple_image, transformation_map = DAModel(image)
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
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
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
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, partition_scene.cameras_extent,
                                                size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # 每500轮保存一次中间外观解耦图像
            if iteration % 100 == 0:
                decouple_image = decouple_image.cpu()
                decouple_image = transforms.ToPILImage()(decouple_image)
                save_dir = os.path.join(partition_scene.model_path, "decouple_images")
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                decouple_image.save(f"{save_dir}/decouple_image_{partition_id}_{viewpoint_cam.uid}_{iteration}.png")

                transformation_map = transformation_map.cpu()
                transformation_map = transforms.ToPILImage()(transformation_map)
                transformation_map.save(
                    f"{save_dir}/transformation_map_{partition_id}_{viewpoint_cam.uid}_{iteration}.png")

                image = image.cpu()
                image = transforms.ToPILImage()(image)
                image.save(f"{save_dir}/render_image_{partition_id}_{viewpoint_cam.uid}_{iteration}.png")

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           partition_scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    synchronizer.wait()


def pull_down(global_W, local_Ws, n_workers):
    # pull down global model to local
    for rank in range(n_workers):
        for name, value in local_Ws[rank].items():
            local_Ws[rank][name].data = global_W[name].data.clone()


def aggregate(global_W, local_Ws, n_workers):
    # init the global model
    for name, value in global_W.items():
        global_W[name].data = torch.zeros_like(value)

    for rank in range(n_workers):
        for name, value in local_Ws[rank].items():
            global_W[name].data += value.data

    for name in local_Ws[rank].keys():  # 这里只是用于取出模型的参数名而以，因此不用纠结于是第几个local_Ws
        global_W[name].data /= n_workers


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    tb_writer = prepare_output_and_logger(dataset)
    # DAModel = DecoupleAppearanceModel().to(dataset.data_device)  # 定义外观解耦模型
    big_scene = BigScene(dataset)  # 这段代码整个都是加载数据集，同时包含高斯模型参数的加载
    # DAM_optimizer, DAM_scheduler = DAModel.optimize(DAModel)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=dataset.data_device)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    total_partition = len(big_scene.partition_data)  # 分区个数
    synchronizer = mp.Barrier(total_partition)
    local_models = [DecoupleAppearanceModel().to(dataset.data_device) for i in range(total_partition)]
    global_model = DecoupleAppearanceModel().to(dataset.data_device)
    local_Ws = [{key: value for key, value in local_models[i].named_parameters()} for i in range(total_partition)]
    global_W = {key: value for key, value in global_model.named_parameters()}

    for partition_id in range(total_partition):
        pull_down(global_W, local_Ws, total_partition)

        process_list = []
        for partition_id in range(total_partition):
            gaussians = GaussianModel(dataset)
            partition_scene = PartitionScene(dataset, gaussians, partition_id, big_scene.partition_data[partition_id],
                                             big_scene.cameras_extent)
            gaussians.training_setup(opt)

            first_iter = 0
            progress_bar = tqdm(range(first_iter, opt.iterations), desc=f"Training progress partition:{partition_id}")
            first_iter += 1
            # 执行训练循环

            p = mp.Process(target=train_partition, args=(opt, first_iter, iter_start, iter_end,
                        progress_bar, tb_writer, testing_iterations, saving_iterations,
                        partition_scene, debug_from, pipe, dataset, background,
                        gaussians, local_models[partition_id], partition_id, checkpoint_iterations,
                        synchronizer))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()

    aggregate(global_W, local_Ws, total_partition)

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
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: PartitionScene,
                    renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            # {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train',
             'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                         range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
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

    # train.py脚本显式参数
    parser.add_argument("--ip", type=str, default='127.0.0.1')  # 启动GUI服务器的IP地址，默认为127.0.0.1。
    parser.add_argument("--port", type=int, default=6009)  # 用于GUI服务器的端口，默认为6009。
    parser.add_argument("--debug_from", type=int, default=-1)  # 调试缓慢。您可以指定一个迭代(从0开始)，之后上述调试变为活动状态。
    parser.add_argument("--detect_anomaly", default=False)  #
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[7_000, 30_000])  # 训练脚本在测试集上计算L1和PSNR的间隔迭代，默认为7000 30000。
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[7_000, 30_000, 60_000])  # 训练脚本保存高斯模型的空格分隔迭代，默认为7000 30000 <迭代>。
    parser.add_argument("--quiet", default=False)  # 标记以省略写入标准输出管道的任何文本。
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])  # 空格分隔的迭代，在其中存储稍后继续的检查点，保存在模型目录中。
    parser.add_argument("--start_checkpoint", type=str, default=None)  # 路径保存检查点继续训练。
    args = parser.parse_args()
    args.save_iterations.append(args.iterations)
    args.source_path = os.path.abspath(args.source_path)  # 将相对路径转换为绝对路径

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
