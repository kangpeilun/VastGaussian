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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx,
										int deg,
										int max_coeffs,
										const glm::vec3 *means,
										glm::vec3 campos,
										const float *shs,
										bool *clamped) {
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0) {
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1) {
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2) {
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3 &mean,
							   float focal_x,
							   float focal_y,
							   float tan_fovx,
							   float tan_fovy,
							   const float *cov3D,
							   const float *viewmatrix) {
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return {float(cov[0][0]), float(cov[0][1]), float(cov[1][1])};
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float *cov3D) {
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
							   const float *orig_points,
							   const glm::vec3 *scales,
							   const float scale_modifier,
							   const glm::vec4 *rotations,
							   const float *opacities,
							   const float *shs,
							   bool *clamped,
							   const float *cov3D_precomp,
							   const float *colors_precomp,
							   const float *viewmatrix,
							   const float *projmatrix,
							   const glm::vec3 *cam_pos,
							   const int W, int H,
							   const float tan_fovx, float tan_fovy,
							   const float focal_x, float focal_y,
							   int *radii,
							   float2 *points_xy_image,
							   float *depths,
							   float *cov3Ds,
							   float *rgb,
							   float4 *conic_opacity,
							   const dim3 grid,
							   uint32_t *tiles_touched,
							   bool prefiltered) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	// 初始化半径和触摸贴图为0。如果这个不改变，这个高斯函数将不会被进一步处理。
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside. 在附近进行扑杀，在室外停止。
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting 投影变换点
	float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};  // 得到椭圆的中心点
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);  // 根据投影矩阵将3D的点投影到2D平面
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters.
	// 如果3D协方差矩阵是预先计算的，使用它，否则从缩放和旋转参数计算。
	const float *cov3D;
	if (cov3D_precomp != nullptr) {
		cov3D = cov3D_precomp + idx * 6;
	} else {
		// 计算椭球的x y z三个轴的长度
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix 计算二维屏幕空间协方差矩阵
	// 计算投影出来的2D椭圆的形状会是什么样的，3D->2D
	// 通过一个2x2的矩阵来记录椭圆的长轴和短轴
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)  反协方差(EWA算法)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles.
	// 计算屏幕空间的范围(通过查找二维协方差矩阵的特征值)。使用extent来计算高斯函数与之重叠的屏幕空间tile的边界矩形。如果矩形覆盖0块，退出。
	// 应用特征值分解计算椭圆的长短轴长，也就是计算一个一元二次方程，会有两个根
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));  // 加法得到的根，对应长轴半径
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));  // 减法得到的根，对应短轴半径
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));  // 乘以3，目的是将高斯分布正负3个标准差的值都包含进去，并作为近似的圆的半径
	float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);  // 计算每个圆覆盖了哪些像素，使用tail-based策略的话，也就是计算这个圆覆盖了哪些格子
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	// 如果颜色已经预先计算，使用它们，否则将球面谐波系数转换为RGB颜色。
	if (colors_precomp == nullptr) {
		// 每个3D高斯，从不同的角度去看，都会有不同的颜色
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3 *)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	// 为接下来的步骤存储一些有用的助手数据。
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = {conic.x, conic.y, conic.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// 主要栅格化方法。协作工作在一个块上，每个线程处理一个像素。在获取和栅格化数据之间交替。
template<uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2 *__restrict__ ranges,
	const uint32_t *__restrict__ point_list,
	int W, int H,
	const float2 *__restrict__ points_xy_image,
	const float *__restrict__ features,
	const float4 *__restrict__ conic_opacity,
	float *__restrict__ final_T,
	uint32_t *__restrict__ n_contrib,
	const float *__restrict__ bg_color,
	float *__restrict__ out_color) {
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = {(float)pix.x, (float)pix.y};

	// Check if this thread is associated with a valid pixel or outside.
	// 检查此线程是否与有效像素或外部相关联。
	// 因为tail-based会将图片划分成多个大小为16x16的网格，如果图像的尺寸不能被16整除的话，
	// 需要对不能被整除的部分额外添加一个网格补全，而多补出来的部分其实是没有像素对应的，因此
	// 在渲染的时候，多补的部分是不需要渲染的
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	// 完成的线程可以帮助抓取，但不能栅格化
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	// 为集体获取的数据批分配存储空间。
	// 这里是将每个Tail中排序后的高斯的ID和坐标存入shared memory中
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables  初始化辅助变量
	float T = 1.0f;  // T表示透过率，随着透过的高斯的数量的增加，透过率会越来越小
	uint32_t contributor = 0;  // 表示像素总共经过多少个高斯的叠加
	uint32_t last_contributor = 0;
	float C[CHANNELS] = {0};  // 记录最后渲染出来的颜色

	// Iterate over batches until all done or range is complete
	// 对批处理进行迭代，直到全部完成或范围完成
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
		// End if entire block votes that it is done rasterizing
		// 如果整个块投票决定栅格化完成，则结束
		int num_done = __syncthreads_count(done);  // 对每个像素是否渲染完成进行计数
		if (num_done == BLOCK_SIZE)  // 当一个Block中所有的像素都渲染完成，则结束
			break;

		// Collectively fetch per-Gaussian data from global to shared
		// 集体获取从全局到共享的每高斯数据
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y) {
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch 迭代当前批处理
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
			// Keep track of current position in range 跟踪当前的位置在范围内
			contributor++;

			// Resample using conic matrix (cf. "Surface  使用二次矩阵重新采样
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];  // 取得一个待渲染的像素的坐标
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};  // 计算像素与高斯几何中心的距离
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			// 式(2)，由三维高斯溅射纸得到。通过乘以高斯不透明度及其从平均值的指数衰减来获得alpha。避免数值不稳定(见论文附录)。
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f) {
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			// 跟踪最后一个范围条目以更新该像素。
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	// 所有处理有效像素的线程将其最终渲染数据写入帧和辅助缓冲区。
	if (inside) {
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2 *ranges,
	const uint32_t *point_list,
	int W, int H,
	const float2 *means2D,
	const float *colors,
	const float4 *conic_opacity,
	float *final_T,
	uint32_t *n_contrib,
	const float *bg_color,
	float *out_color) {
	renderCUDA<NUM_CHANNELS> <<< grid, block >>>(
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
						 const float *means3D,
						 const glm::vec3 *scales,
						 const float scale_modifier,
						 const glm::vec4 *rotations,
						 const float *opacities,
						 const float *shs,
						 bool *clamped,
						 const float *cov3D_precomp,
						 const float *colors_precomp,
						 const float *viewmatrix,
						 const float *projmatrix,
						 const glm::vec3 *cam_pos,
						 const int W, int H,
						 const float focal_x, float focal_y,
						 const float tan_fovx, float tan_fovy,
						 int *radii,
						 float2 *means2D,
						 float *depths,
						 float *cov3Ds,
						 float *rgb,
						 float4 *conic_opacity,
						 const dim3 grid,
						 uint32_t *tiles_touched,
						 bool prefiltered) {
	preprocessCUDA<NUM_CHANNELS> <<< (P + 255) / 256, 256 >>> (
		P, D, M,
			means3D,
			scales,
			scale_modifier,
			rotations,
			opacities,
			shs,
			clamped,
			cov3D_precomp,
			colors_precomp,
			viewmatrix,
			projmatrix,
			cam_pos,
			W, H,
			tan_fovx, tan_fovy,
			focal_x, focal_y,
			radii,
			means2D,
			depths,
			cov3Ds,
			rgb,
			conic_opacity,
			grid,
			tiles_touched,
			prefiltered
	);
}