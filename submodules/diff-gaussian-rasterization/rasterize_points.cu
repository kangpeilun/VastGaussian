/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
// TODO: 调试时先去掉 <torch/extension.h>
//#include <torch/extension.h>  // 在用pip进行安装时，再引入这一项
#include <torch/torch.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "rasterize_points.h"
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t) {
    // 这个函数可以用来动态调整张量的大小，使其能够容纳更多的元素。
    // 这个函数接受一个torch::Tensor类型的引用作为参数，并返回一个std::function对象。
    // 该std::function对象接受一个size_t类型的参数N，表示要调整的新的张量大小。
    // 在lambda函数内部，首先通过调用t.resize_({(long long) N})来调整张量的大小。
    // 然后，使用t.contiguous().data_ptr()获取张量的数据指针，并将其转换为char *类型。
    // 最后，返回这个char *类型的指针，作为重置张量大小后的结果。
    auto lambda = [&t](size_t N) {
        t.resize_({(long long) N});
        return reinterpret_cast<char *>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
        const torch::Tensor &background,
        const torch::Tensor &means3D,
        const torch::Tensor &colors,
        const torch::Tensor &opacity,
        const torch::Tensor &scales,
        const torch::Tensor &rotations,
        const float scale_modifier,
        const torch::Tensor &cov3D_precomp,
        const torch::Tensor &viewmatrix,
        const torch::Tensor &projmatrix,
        const float tan_fovx,
        const float tan_fovy,
        const int image_height,
        const int image_width,
        const torch::Tensor &sh,
        const int degree,
        const torch::Tensor &campos,
        const bool prefiltered,
        const bool debug) {
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }

    const int P = means3D.size(0);  // Point_num  点云的数量
    const int H = image_height;  // 545
    const int W = image_width;  // 980

    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);
    /*torch::full是PyTorch库中的一个函数，用于创建一个填充指定尺寸和数据类型的新张量
        size: 张量的尺寸，以torch::Size形式给出。
        value: 张量的初始值。
        dtype: 张量的数据类型。默认为torch::kFloat32。
        device: 张量的设备。默认为None，表示张量将在CPU上创建。
        pin_memory: 是否将张量设置为pinned内存。默认为None，表示不设置。
     */
    torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);  // [3, H, W] 定义输出图片的尺寸
    torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));  // [Point_num,]

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);  // torch::kByte常量表示byte数据类型，即8位无符号整数。
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));  // torch::empty是一个PyTorch库中的函数，用于创建一个大小未指定的新张量。它通常用于初始化一个新的张量，该张量的大小将在稍后的某个时间点分配。
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);

    int rendered = 0;
    if (P != 0) {
        int M = 0;
        if (sh.size(0) != 0) {
            M = sh.size(1);
        }

        rendered = CudaRasterizer::Rasterizer::forward(
                geomFunc,
                binningFunc,
                imgFunc,
                P, degree, M,
                background.contiguous().data<float>(),
                W, H,
                means3D.contiguous().data<float>(),
                sh.contiguous().data_ptr<float>(),
                colors.contiguous().data<float>(),
                opacity.contiguous().data<float>(),
                scales.contiguous().data_ptr<float>(),
                scale_modifier,
                rotations.contiguous().data_ptr<float>(),
                cov3D_precomp.contiguous().data<float>(),
                viewmatrix.contiguous().data<float>(),
                projmatrix.contiguous().data<float>(),
                campos.contiguous().data<float>(),
                tan_fovx,
                tan_fovy,
                prefiltered,
                out_color.contiguous().data<float>(),
                radii.contiguous().data<int>(),
                debug);
    }

//     std::cout << "out_color" << out_color.sizes() << std::endl;
//     std::cout << "Rendered " << rendered << " points" << std::endl;
//     std::cout << "radii" << out_color.sizes() << std::endl;
//     std::cout << "geomBuffer" << out_color.sizes() << std::endl;
//     std::cout << "binningBuffer" << out_color.sizes() << std::endl;
//     std::cout << "imgBuffer" << out_color.sizes() << std::endl;

    return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
        const torch::Tensor &background,
        const torch::Tensor &means3D,
        const torch::Tensor &radii,
        const torch::Tensor &colors,
        const torch::Tensor &scales,
        const torch::Tensor &rotations,
        const float scale_modifier,
        const torch::Tensor &cov3D_precomp,
        const torch::Tensor &viewmatrix,
        const torch::Tensor &projmatrix,
        const float tan_fovx,
        const float tan_fovy,
        const torch::Tensor &dL_dout_color,
        const torch::Tensor &sh,
        const int degree,
        const torch::Tensor &campos,
        const torch::Tensor &geomBuffer,
        const int R,
        const torch::Tensor &binningBuffer,
        const torch::Tensor &imageBuffer,
        const bool debug) {
    const int P = means3D.size(0);
    const int H = dL_dout_color.size(1);
    const int W = dL_dout_color.size(2);

    int M = 0;
    if (sh.size(0) != 0) {
        M = sh.size(1);
    }

    torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
    torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
    torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
    torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
    torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
    torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

    if (P != 0) {
        CudaRasterizer::Rasterizer::backward(P, degree, M, R,
                                             background.contiguous().data<float>(),
                                             W, H,
                                             means3D.contiguous().data<float>(),
                                             sh.contiguous().data<float>(),
                                             colors.contiguous().data<float>(),
                                             scales.data_ptr<float>(),
                                             scale_modifier,
                                             rotations.data_ptr<float>(),
                                             cov3D_precomp.contiguous().data<float>(),
                                             viewmatrix.contiguous().data<float>(),
                                             projmatrix.contiguous().data<float>(),
                                             campos.contiguous().data<float>(),
                                             tan_fovx,
                                             tan_fovy,
                                             radii.contiguous().data<int>(),
                                             reinterpret_cast<char *>(geomBuffer.contiguous().data_ptr()),
                                             reinterpret_cast<char *>(binningBuffer.contiguous().data_ptr()),
                                             reinterpret_cast<char *>(imageBuffer.contiguous().data_ptr()),
                                             dL_dout_color.contiguous().data<float>(),
                                             dL_dmeans2D.contiguous().data<float>(),
                                             dL_dconic.contiguous().data<float>(),
                                             dL_dopacity.contiguous().data<float>(),
                                             dL_dcolors.contiguous().data<float>(),
                                             dL_dmeans3D.contiguous().data<float>(),
                                             dL_dcov3D.contiguous().data<float>(),
                                             dL_dsh.contiguous().data<float>(),
                                             dL_dscales.contiguous().data<float>(),
                                             dL_drotations.contiguous().data<float>(),
                                             debug);
    }

    return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales,
                           dL_drotations);
}

torch::Tensor markVisible(
        torch::Tensor &means3D,
        torch::Tensor &viewmatrix,
        torch::Tensor &projmatrix) {
    const int P = means3D.size(0);

    torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

    if (P != 0) {
        CudaRasterizer::Rasterizer::markVisible(P,
                                                means3D.contiguous().data<float>(),
                                                viewmatrix.contiguous().data<float>(),
                                                projmatrix.contiguous().data<float>(),
                                                present.contiguous().data<bool>());
    }

    return present;
}