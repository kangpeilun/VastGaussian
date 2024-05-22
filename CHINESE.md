# VastGaussian
This it the [ENGLISH](ENGLISH.md) Version.

![img.png](image/img_.png)

这是`VastGaussian: Vast 3D Gaussians for Large Scene Reconstruction`的非官方实现，因为是第一次从头复现完整的代码，因此代码可能会出现一些错误，并且代码的写法和一些高手相比可能会显得有些幼稚，缺少一些工程上的技巧。
不过我也迈出了自己的第一步，因为我在网络上找不到任何关于VastGaussian的任何实现，于是我进行了一下尝试。

如果大家在使用过程中有任何代码修改方面的经验和反馈，反应联系我，或者简单的提出你的Issue：
> Email: 374774222@qq.com
> 
> QQ: 374774222
> 
> WeChat: k374774222

## ToDo List
- [x] 实现Camera-position-based region division
- [x] 实现Position-based data selection
- [x] 实现Visibility-based camera selection
- [x] 实现Coverage-based point selection
- [x] 实现Decoupled Appearance Modeling
- [x] 实现Seamless Merging
- [ ] 实现将点云进行division后，m*n个region在单GPU上的并行训练
- [ ] 在UrbanScene3D和Mill-19数据集上进行实验

## 说明

1. 我在原始的3DGS上进行了修改，首先我将3DGS的超参数从`arguments/__init__.py`中摘取了出来放在了`arguments/parameters.py`文件里，更加方便阅读和理解超参的含义
2. 为了不改变3DGS原本的目录结构，我新添加了一个`VastGaussian_scene`用于存放VastGaussian的模块，其中一部分代码我调用了`scene`文件夹中已有的函数，同时为了解决`import`的错误，我将Scene类移动到了datasets.py文件夹里

<div align="center">
    <img src=image/img2.png align="center"> 
    <img src=image/img_1.png align="center">
</div>

3. 文件的命名与论文中提到的方法保持一致，方便阅读

> `datasets.py` 我对3DGS中的Scene类进行了重写，分成BigScene和PartitionScene，前者表示原始的场景BigScene，后者表示经过Partition后的各个小场景PartitionScene
>
> `data_partition.py` 数据分区，对应论文 `Progressive Data Partitioning`
>
> <img src=image/img_3.png align="center" width=800>
>
> `decouple_appearance_model.py` 外观解耦模块，对应论文 `Decoupled Appearance Modeling`
>
> <div align="center">
>     <img src=image/img.png align="center" height=400>
>     <img src=image/img_2.png align="center" width=400>
> </div> 
>
> `graham_scan.py` 凸包计算，用于在实现Visibility-based camera selection时，将partition后的立方体投影到相机平面上，并计算投影区域与图片区域的交集
>
> `seamless_merging.py` 无缝合并，对应论文 `Seamless Merging`，将各个PartitionScene合并成BigScene

4. 我新增了一个`train_vast.py`文件，对训练VastGaussian的过程进行了修改，如果想对原始的3DGS进行训练，请使用`train.py`

5. 论文中提到进行`曼哈顿世界对齐，使世界坐标的y轴垂直于地平面`，我在询问高人才知道，这个东西可以使用CloudCompare软件进行手动调整，其大体过程就是将点云所在的区域的包围盒边界调整到与点云区域的整体朝向保持平行
> 比如下图中的点云原本是倾斜的，经过调整好变成水平和垂直的，高人说是曼哈顿世界对其是大尺度三维重建的基本操作(方便进行partition)，哈哈
>
> <div align="center">
>     <img src=image/img_4.png height=300>
>     <img src=image/img_5.png height=300>
> </div>
>
> 我在实现过程中使用的是3DGS提供的小范围数据进行的测试，较大的数据本机跑不了，大范围的数据根据论文的说明至少要32G显存
7. 在实现过程中，在论文中的一些操作，作者并没有很明确的说明细节，因此一些实现是根据我的猜测和理解去完成的，也因此我的实现可能会有一些bug，并且有些实现在高手看来可能有些蠢，如果大家在使用过程中发现有问题，请及时联系我，一起进步

## 使用

1. 数据格式和3DGS一样，同时训练的命令也和3DGS基本一样，我没有进行什么太多个性化的修改，你可以参考下面的命令(更多的参数请参考`arguments/parameters.py`):
```python
python train_vast.py -s output/dataset --exp_name test
```

## 数据集
1. `Urbanscene3D`: https://github.com/Linxius/UrbanScene3D

2. `Mill-19`: https://opendatalab.com/OpenDataLab/Mill_19/tree/main/raw

3. 测试数据: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip