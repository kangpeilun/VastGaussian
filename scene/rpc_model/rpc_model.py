# -*- coding: utf-8 -*- 
# @Author: Kang Peilun
# @Email: 374774222@qq.com
# @File: rpc_model.py
# @Project: Geo3DGS
# @Time: 2023/12/22 上午11:11
# @Des:

# Myron Brown, 2018
# RPC00B specification is described in http://geotiff.maptools.org/rpc_prop.html

import math
import numpy as np
from osgeo import gdal
from math import sqrt
from utm import *
import time


class RPC:

    # Initialization must read metadata 初始化必须读取元数据
    def __init__(self, img_name):
        self.read_metadata(img_name)

    # Read metadata
    def read_metadata(self, img_name):

        # Read the metadata
        dataset = gdal.Open(img_name, gdal.GA_ReadOnly)
        metadata = dataset.GetMetadata()
        rpc_data = dataset.GetMetadata('RPC')

        # read image and get size
        img = dataset.ReadAsArray()
        self.rows = img.shape[1]
        self.columns = img.shape[2]
        img = None

        # Extract RPC metadata fields
        self.lon_off = float(rpc_data['LONG_OFF'])
        self.lon_scale = float(rpc_data['LONG_SCALE'])
        self.lat_off = float(rpc_data['LAT_OFF'])
        self.lat_scale = float(rpc_data['LAT_SCALE'])
        self.height_off = float(rpc_data['HEIGHT_OFF'])
        self.height_scale = float(rpc_data['HEIGHT_SCALE'])
        self.line_off = float(rpc_data['LINE_OFF'])
        self.line_scale = float(rpc_data['LINE_SCALE'])
        self.samp_off = float(rpc_data['SAMP_OFF'])
        self.samp_scale = float(rpc_data['SAMP_SCALE'])
        self.line_num_coeff = np.asarray(rpc_data['LINE_NUM_COEFF'].split(), dtype=np.float64)
        self.line_den_coeff = np.asarray(rpc_data['LINE_DEN_COEFF'].split(), dtype=np.float64)
        self.samp_num_coeff = np.asarray(rpc_data['SAMP_NUM_COEFF'].split(), dtype=np.float64)
        self.samp_den_coeff = np.asarray(rpc_data['SAMP_DEN_COEFF'].split(), dtype=np.float64)

        # Close the dataset
        dataset = None

    # Forward project normalized WGS84 array to image coordinates
    # Rows and columns are computed with separate calls and coefficient arrays
    # Not currently checking the denominator for zero
    # 正向投影标准化的WGS84数组到图像坐标行和列使用单独的调用和系数数组计算当前未检查分母是否为零
    def forward_array_normalized(self, num_coeff, den_coeff, X, Y, Z):
        count = len(X)
        ones = np.full(count, 1.0)
        monomial = np.array([ones, X, Y, Z, X * Y, X * Z, Y * Z, \
                             X * X, Y * Y, Z * Z, X * Y * Z, \
                             X * X * X, X * Y * Y, X * Z * Z, X * X * Y, \
                             Y * Y * Y, Y * Z * Z, X * X * Z, Y * Y * Z, Z * Z * Z])
        num_array = np.transpose(np.tile(num_coeff, (count, 1)))
        den_array = np.transpose(np.tile(den_coeff, (count, 1)))
        num = sum(num_array * monomial)
        den = sum(den_array * monomial)
        pixel_coordinates = num / den
        return pixel_coordinates

    # Forward project WGS84 array to image coordinates 将WGS84阵列正向投影到图像坐标
    # 地理坐标到图像坐标
    def forward_array(self, X, Y, Z):

        # Normalize the WGS84 coordinates in place
        X = (X - self.lon_off) / self.lon_scale
        Y = (Y - self.lat_off) / self.lat_scale
        Z = (Z - self.height_off) / self.height_scale

        # Forward project normalized WGS84 array to image coordinates
        cols = self.forward_array_normalized(self.samp_num_coeff, self.samp_den_coeff, X, Y, Z)
        rows = self.forward_array_normalized(self.line_num_coeff, self.line_den_coeff, X, Y, Z)

        # Normalize the image coordinates
        cols = (cols * self.samp_scale) + self.samp_off
        rows = (rows * self.line_scale) + self.line_off

        # Renormalize the WGS84 coordinates
        X = (X * self.lon_scale) + self.lon_off
        Y = (Y * self.lat_scale) + self.lat_off

        return cols, rows

    # Forward project normalized WGS84 scalar to image coordinates 正向投影归一化WGS84标量到图像坐标 (得到归一化后的图像坐标)
    # Rows and columns are computed with separate calls and coefficient arrays 行和列是使用单独的调用和系数数组计算的
    # Not currently checking the denominator for zero 当前未检查分母是否为零
    def forward_normalized(self, num_coeff, den_coeff, X, Y, Z):
        monomial = [1, X, Y, Z, X * Y, X * Z, Y * Z, \
                    X * X, Y * Y, Z * Z, X * Y * Z, \
                    X * X * X, X * Y * Y, X * Z * Z, X * X * Y, \
                    Y * Y * Y, Y * Z * Z, X * X * Z, Y * Y * Z, Z * Z * Z]
        num = sum(num_coeff * monomial)
        den = sum(den_coeff * monomial)
        pixel_coordinate = num / den
        return pixel_coordinate

    # Forward project WGS84 scalar to image coordinates
    # 将WGS84标量正向投影到图像坐标
    def forward(self, X, Y, Z):

        # Normalize the WGS84 coordinates in place
        X = (X - self.lon_off) / self.lon_scale
        Y = (Y - self.lat_off) / self.lat_scale
        Z = (Z - self.height_off) / self.height_scale

        # Forward project normalized WGS84 scalar to image coordinates
        col = self.forward_normalized(self.samp_num_coeff, self.samp_den_coeff, X, Y, Z)
        row = self.forward_normalized(self.line_num_coeff, self.line_den_coeff, X, Y, Z)

        # Normalize the image coordinates
        col = (col * self.samp_scale) + self.samp_off
        row = (row * self.line_scale) + self.line_off

        # Renormalize the WGS84 coordinates
        X = (X * self.lon_scale) + self.lon_off
        Y = (Y * self.lat_scale) + self.lat_off

        return col, row

    # Compute approximate world coordinates at center of image 计算图像中心的近似世界坐标
    def approximate_wgs84(self):

        # get the center image coordinate
        print('Pixels (rows, columns): ', self.rows, self.columns)
        row = self.rows / 2.0
        col = self.columns / 2.0

        # find approximate world coordinate in center of image
        # rpc world coordinate offset is for full source image, not the tiled image
        # rpc image coordinate offset is for the image tile
        # without a dsm, assume flat earth
        lat = self.lat_off
        lon = self.lon_off
        height = self.height_off
        delta = (0.25 / 111000.0)  # about 25 centimeters
        rms = math.inf
        for k in range(1000):
            # perturb coordinates to get an update
            i1, j1 = self.forward(lon, lat, height)
            i2, j2 = self.forward(lon + delta, lat, height)
            i3, j3 = self.forward(lon, lat + delta, height)
            e1 = sqrt((i1 - col) * (i1 - col) + (j1 - row) * (j1 - row))
            e2 = sqrt((i2 - col) * (i2 - col) + (j2 - row) * (j2 - row))
            e3 = sqrt((i3 - col) * (i3 - col) + (j3 - row) * (j3 - row))

            # apply update and check result
            # when error stops decreasing, it's close enough to stop
            tlon = lon + (np.sign(e1 - e2) * abs(e2) * delta / 2.0)
            tlat = lat + (np.sign(e1 - e3) * abs(e3) * delta / 2.0)
            i1, j1 = self.forward(tlon, tlat, height)
            e1 = sqrt((i1 - col) * (i1 - col) + (j1 - row) * (j1 - row))
            if (e1 > rms):
                break
            lon = tlon
            lat = tlat
            rms = e1

        # print the solution
        i1, j1 = self.forward(lon, lat, height)
        e1 = sqrt((i1 - col) * (i1 - col) + (j1 - row) * (j1 - row))
        print('Coordinates: ', lat, lon, height)
        print('RMS Error (pixels): ', e1, ', converged in', k, 'iterations')

        return lat, lon, height

    # Approximate RPC with a local 3x4 matrix projection 具有局部3x4矩阵投影的近似RPC
    # Default range is approximately 1km horizontal and 500m vertical 默认范围为水平方向约1公里，垂直方向约500米
    # The clat, clon, zc values are image center UTM coordinates that we
    # pass in to make sure they're the same value for all images
    # clat、clon和zc值是图像中心的UTM坐标，我们输入这些坐标以确保它们对所有图像都是相同的值
    def to_matrix(self, clat, clon, zc, x_km=1.0, z_km=0.5, num_samples=100):

        # get approximate image center coordinate 获取近似图像中心坐标
        xc, yc, zone_number, zone_letter = wgs84_to_utm(clat, clon)

        # sample local world coordinates around the center coordinate 围绕中心坐标采样局部世界坐标
        np.random.seed(0)
        dlat = dlon = (x_km / 2.0) / 111.0
        dheight = (z_km / 2.0) * 1000.0
        lat = np.random.uniform(clat - dlat, clat + dlat, num_samples)
        lon = np.random.uniform(clon - dlon, clon + dlon, num_samples)
        z = np.random.uniform(zc - dheight, zc + dheight, num_samples)

        # project world coordinates to image coordinates 将世界坐标投影到图像坐标
        i, j = self.forward_array(lon, lat, z)

        # project to UTM coordinates 投影到UTM坐标
        x, y, zone_number, zone_letter = wgs84_to_utm_array(lat, lon)

        # compute XYZ means and subtract them 计算XYZ平均值并将其相减
        ic = np.average(i)
        jc = np.average(j)
        x = x - xc
        y = y - yc
        z = z - zc
        i = i - ic
        j = j - jc
        print('XYZIJ Centers: ', xc, yc, zc, ic, jc)

        # solve Normal equations 求解法线方程
        start_time = time.time()
        M = np.zeros((num_samples * 3, 15))
        print('Normal equations matrix size = ', M.shape)
        m = 0
        for k in range(num_samples):
            # populate the i terms 填充i项
            M[m][0] = x[k];
            M[m][1] = y[k];
            M[m][2] = z[k];
            M[m][3] = 1.0
            M[m][4] = 0.0;
            M[m][5] = 0.0;
            M[m][6] = 0.0;
            M[m][7] = 0.0
            M[m][8] = 0.0;
            M[m][9] = 0.0;
            M[m][10] = 0.0;
            M[m][11] = 0.0
            M[m][12] = -x[k] * i[k]
            M[m][13] = -y[k] * i[k]
            M[m][14] = -z[k] * i[k]
            m = m + 1

            # populate the j terms 填充j项
            M[m][0] = 0.0;
            M[m][1] = 0.0;
            M[m][2] = 0.0;
            M[m][3] = 0.0
            M[m][4] = x[k];
            M[m][5] = y[k];
            M[m][6] = z[k];
            M[m][7] = 1.0
            M[m][8] = 0.0;
            M[m][9] = 0.0;
            M[m][10] = 0.0;
            M[m][11] = 0.0
            M[m][12] = -x[k] * j[k]
            M[m][13] = -y[k] * j[k]
            M[m][14] = -z[k] * j[k]
            m = m + 1

            # populate the k terms (k is zero)
            M[m][0] = 0.0;
            M[m][1] = 0.0;
            M[m][2] = 0.0;
            M[m][3] = 0.0
            M[m][4] = 0.0;
            M[m][5] = 0.0;
            M[m][6] = 0.0;
            M[m][7] = 0.0
            M[m][8] = x[k];
            M[m][9] = y[k];
            M[m][10] = z[k];
            M[m][11] = 1.0
            m = m + 1

        elapsed_total = (time.time() - start_time)
        print('Populate matrix time = ', elapsed_total)

        start_time = time.time()
        invM = np.linalg.pinv(M)
        elapsed_total = (time.time() - start_time)
        print('Invert matrix time = ', elapsed_total)

        b = np.zeros((num_samples * 3))
        m = 0
        for k in range(num_samples):
            b[m] = i[k]
            b[m + 1] = j[k]
            m = m + 3
        a = np.matmul(invM, b)
        print('Rotation matrix size = ', a.shape)  # 旋转矩阵大小
        # [ii] = [ (00) (01) (02) (03) ][x]
        # [jj] = [ (04) (05) (06) (07) ][y]
        # [rr] = [ (08) (09) (10) (11) ][z]
        # [qq] = [ (12) (13) (14)  1.0 ][1]

        # compute residual errors in for loop 计算for循环中的残差
        rms = 0.0
        for k in range(num_samples):
            q = a[12] * x[k] + a[13] * y[k] + a[14] * z[k] + 1.0
            newi = (a[0] * x[k] + a[1] * y[k] + a[2] * z[k] + a[3]) / q
            newj = (a[4] * x[k] + a[5] * y[k] + a[6] * z[k] + a[7]) / q
            rms += ((i[k] - newi) * (i[k] - newi) + (j[k] - newj) * (j[k] - newj))
        rms /= num_samples
        print('RMS error (pixels) = ', rms)

        # extract terms for 3x4 projection matrix  3x4投影矩阵的提取项
        R = np.ones((3, 4))
        R[0, 0] = a[0];
        R[0, 1] = a[1];
        R[0, 2] = a[2];
        R[0, 3] = a[3]
        R[1, 0] = a[4];
        R[1, 1] = a[5];
        R[1, 2] = a[6];
        R[1, 3] = a[7]
        R[2, 0] = a[12];
        R[2, 1] = a[13];
        R[2, 2] = a[14];
        R[2, 3] = 1.0
        print('3x4 Matrix = ')
        print(R)

        # compute residual errors using matrix operation  使用矩阵运算计算残差
        b = np.array((x, y, z, np.ones(num_samples)))
        a = np.array((i, j, np.ones(num_samples)))
        print(b.shape)
        print(a.shape)
        u = np.matmul(R, b)
        u[:] /= u[2]
        diff = np.sqrt(u * u) / num_samples
        print('RMS error (pixels) = ', rms)

        return R, rms, ic, jc
