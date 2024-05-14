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

find_package(PythonInterp REQUIRED)
cmake_minimum_required(VERSION 3.20)

project(DiffRast LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CUDA_NVCC_FLAGS -g;-G)

set(CUDA_TOOLKIT_ROOT_DIR /home/kpl/software/cuda-11.8/cuda)
set(CMAKE_PREFIX_PATH /home/kpl/software/Anaconda/Anaconda/envs/torch_cuda/lib/python3.8/site-packages/torch/share/cmake)
find_package(Torch REQUIRED)

add_library(CudaRasterizer STATIC
	cuda_rasterizer/backward.h
	cuda_rasterizer/backward.cu
	cuda_rasterizer/forward.h
	cuda_rasterizer/forward.cu
	cuda_rasterizer/auxiliary.h
	cuda_rasterizer/rasterizer_impl.cu
	cuda_rasterizer/rasterizer_impl.h
	cuda_rasterizer/rasterizer.h
)

add_executable(DiffRast
		main.cpp
		rasterize_points.h
		rasterize_points.cu
		)
target_link_libraries(DiffRast ${TORCH_LIBRARIES})
set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86")

target_link_libraries(DiffRast CudaRasterizer)  # 将库链接到可执行文件
target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_include_directories(CudaRasterizer PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})