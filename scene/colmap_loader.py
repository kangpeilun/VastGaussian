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

import numpy as np
import collections
import struct

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])  # 用具名元组的方式，存放外参参数
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {  # TODO: 已知的相机模型
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),  # 针孔相机模型，适用于无畸变的图像，SIMPLE使用一个焦距参数即f，可以理解为fx=fy
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),    # 针孔相机模型，PINHOLE使用两个焦距参数fx和fy
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    # 旋转向量 转 旋转矩阵
    # 旋转轴/旋转角、旋转矩阵、四元数、李代数都可以表示旋转
    """等价于下面代码
    conda install -c conda-forge quaternion

    import quaternion
    q = np.quaternion(qvec[0], qvec[1], qvec[2], qvec[3])
    rotate_matrix = quaternion.as_rotation_matrix(q)
    """
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):  # 旋转矩阵转四元数
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.  从二进制文件中读取并解包下一个字节。
    :param fid: 表示要读取的文件对象
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.  表示要读取和解包的字节数。只能是{2, 4, 8}的组合和，例如2、6、16、30等。
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.  表示要解包的格式字符序列的列表。格式字符包括{c, e, f, d, h, H, i, I, l, L, q, Q}。
    :param endian_character: Any of {@, =, <, >, !}  表示字节顺序的字符。可以是{@, =, <, >, !}中的任意一个。
    :return: Tuple of read and unpacked values.  读取和解压缩值的元组。
    """
    """
    fid.read(num_bytes)是一个文件对象的方法，用于从文件中读取指定数量的字节数据。
    参数num_bytes表示要读取的字节数。该方法会从文件的当前位置开始读取指定数量的字节，并将读取的数据作为字符串返回。
        如果文件中没有足够的字节可供读取，则返回的字符串可能少于指定的字节数。
    每次调用fid.read(num_bytes)后，文件对象的当前位置会自动向后移动读取的字节数，以便下一次读取操作可以从当前位置继续读取。
    """
    data = fid.read(num_bytes)

    """
    struct.unpack(格式字符串, 要解包的二进制数据)是一个用于解包二进制数据的函数
    格式字符串指定了要按照哪种格式解包二进制数据，并指定了解包后的数据类型和顺序。格式字符串由格式字符和可选的字节顺序字符组成。
    解包的结果将作为一个元组返回，其中包含了按照格式字符串解包后的值。
    格式字符用于指定要解包的数据类型和大小。以下是所有的格式字符及其含义：
            x：跳过一个字节
            b：有符号字节
            B：无符号字节
            h：有符号短整数（2字节）
            H：无符号短整数（2字节）
            i：有符号整数（4字节）
            I：无符号整数（4字节）
            l：有符号长整数（4字节）
            L：无符号长整数（4字节）
            q：有符号长长整数（8字节）
            Q：无符号长长整数（8字节）
            f：单精度浮点数（4字节）
            d：双精度浮点数（8字节）
            s：字符串（需指定长度）
            p：Pascal字符串（需指定长度）
            P：无符号长整数（和机器字长一样）
        常用的字节顺序字符包括：
            <：小端字节顺序，低位字节在前，高位字节在后。例如，整数1234以小端字节顺序存储为[0xD2, 0x04]。
            >：大端字节顺序，高位字节在前，低位字节在后。例如，整数1234以大端字节顺序存储为[0x04, 0xD2]。
            !：网络字节顺序，也称为大端字节顺序，是因特网协议中规定的字节顺序。在网络通信中，数据通常以网络字节顺序传输。
    """
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1


    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                # assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))  # 相机内参
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]  # 获取一共有多少张图片
        for _ in range(num_reg_images):
            """
            idddddddi表示读取 2个有符号整数(共8字节)，7个双精度浮点数(共56字节)，一共64字节
            因此在使用read_next_bytes时要传入的 要读取的字节数 要与格式化字符串的类型 要一致
            """
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]  # 获取图像对应id
            qvec = np.array(binary_image_properties[1:5])  # 获取对应图片的旋转矩阵
            tvec = np.array(binary_image_properties[5:8])  # 获取对应图片的平移向量
            camera_id = binary_image_properties[8]  # 获取图片对应相机的id
            image_name = ""  # 获取图像名称
            current_char = read_next_bytes(fid, 1, "c")[0]  # c表示一个字符，一个字符长度为一个字节
            while current_char != b"\x00":   # look for the ASCII 0 entry  b"\x00"表示读到空字符，这行读取结束。从而获取完整的文件名
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]  # 获取二维图像中关键点的数量
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)  # 获取所有关键点的坐标，以及其是否含有对应的3D点
            """
            x_y_id_s[0::3] 表示从第0个索引开始，每隔3个元素取一个值，这行代码的含义其实是保存 二维图像关键点的x轴坐标
            x_y_id_s[1::3] 这行代码的含义其实是保存 二维图像关键点的y轴坐标
            xys 存放关键点的x,y坐标 [[x1,y1], [x2,y2], ...]
            """
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])  # 存放关键点的x,y坐标
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))  # 存放二维关键点对应的3D点的id
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images  # images里存放所有图片的相机外参


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]  # 获取相机数量
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]  # 相机id
            model_id = camera_properties[1]   # 相机型号id
            model_name = CAMERA_MODEL_IDS[model_id].model_name  # 根据相机型号id得到对应的相机名
            width = camera_properties[2]   # 图片宽度
            height = camera_properties[3]  # 图片高度
            num_params = CAMERA_MODEL_IDS[model_id].num_params  # 获取相机参数个数。PINHOLE适用于畸变图像，使用两个焦距参数fx和fy，PINHOLE模型有4个参数
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)  # 获取相机参数
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

def read_extrinsics_binary_vast(path_to_model_file, lines):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            
            # Read the image name even if it is not needed
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]

            # Read the number of 2D points
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]

            # Read all 2D points data even if it is not needed
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D, format_char_sequence="ddq"*num_points2D)

            if image_name not in lines:
                continue  # Continue to the next image after reading all data for the current image

            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]

            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

def read_intrinsics_binary_vast(path_to_model_file, lines):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_colmap_bin_array(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py

    :param path: path to the colmap binary file.
    :return: nd array with the floating point values in the value
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


if __name__ == '__main__':
    # 相机外参文件路径
    # extrinsics_path = r"E:\Pycharm\3D_Reconstruct\gaussian-splatting\datasets\tandt_db\tandt\train\sparse\0\images.bin"
    # read_extrinsics_binary(extrinsics_path)

    # 相机内参文件路径
    intrinsics_path = r"E:\Pycharm\3D_Reconstruct\gaussian-splatting\datasets\tandt_db\tandt\train\sparse\0\cameras.bin"
    read_intrinsics_binary(intrinsics_path)