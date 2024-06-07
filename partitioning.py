
import os
from arguments.parameters import ModelParams, PipelineParams, OptimizationParams, extract, create_man_rans
from argparse import ArgumentParser
from scene.dataset_readers import sceneLoadTypeCallbacks
from VastGaussian_scene.data_partition import ProgressiveDataPartitioning
from utils.camera_utils import cameraList_from_camInfos_partition

if __name__ == "__main__":
    parser = ArgumentParser(description="Training Script Parameters")
    # 三个模块里的参数
    lp = ModelParams(parser).parse_args()
    op, before_extract_op = extract(lp, OptimizationParams(parser).parse_args())
    pp, before_extract_pp = extract(before_extract_op, PipelineParams(parser).parse_args())
    man_trans = create_man_rans(lp.pos, lp.rot)
    lp.man_trans = man_trans
    # train.py脚本显式参数
    args = parser.parse_args()
    args.source_path = os.path.abspath(args.source_path)  # 将相对路径转换为绝对路径
    
    train_cameras = {}
    scene_info = sceneLoadTypeCallbacks["Partition"](args.source_path, args.images, lp.man_trans)  # 得到一个场景的所有参数信息
    train_cameras = cameraList_from_camInfos_partition(scene_info.train_cameras, 
                                                       image_width = 4608,
                                                       image_height = 3456,
                                                       args = args)
    DataPartitioning = ProgressiveDataPartitioning(scene_info, train_cameras, args.model_path,
                                                   m_region=3, n_region=4)
    partition_result = DataPartitioning.load_partition_data()
    client = 0
    for partition in partition_result:
        camera_info = partition.cameras
        image_name_list = [camera_info[i].camera.image_name + '.jpg' for i in range(len(camera_info))]
        txt_file = f"{args.model_path}/partition_point_cloud/visible/{client}/camera.txt"
        # 打开一个文件用于写入，如果文件不存在则会被创建
        with open(txt_file, 'w') as file:
            # 遍历列表中的每个元素
            for item in image_name_list:
                # 将每个元素写入文件，每个元素占一行
                file.write(f"{item}\n")
        client += 1
    
    
    