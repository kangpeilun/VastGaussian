# VastGaussian
This is [Chinese](CHINESE.md) Version.

![img.png](image/img_.png)

This is `VastGaussian: Vast 3D Gaussians for Large Scene Reconstruction` unofficial implementation, since this is my first time to recreate the complete code from scratch, the code may have some errors, and the code writing may seem a bit naive compared to some experts. Lack of engineering skills. But I got my foot in the door. I couldn't find any implementation of VastGaussian on the web, so I gave it a try.

If you have any experiences and feedback on any code changes, feel free to contact me, or simply raise an Issue :grinning::

> Email: 374774222@qq.com
> 
> QQ: 374774222
> 
> WeChat: k374774222

## ToDo List
- [x] ~~Camera-position-based region division is implemented~~

- [x] ~~Position-based data selection is implemented~~

- [x] ~~Visibility-based camera selection is implemented~~

- [x] ~~Coverage-based point selection is implemented~~

- [x] ~~Decoupled Appearance Modeling is implemented~~

- [x] ~~Seamless Merging is implemented~~

- [x] ~~For non-standard scenes by manual Manhattan alignment~~

- [ ] Parallel training of $m\times n$ regions on a single GPU is implemented after dividing the point cloud

- [ ] Experiments are carried out on UrbanScene3D and Mill-19 datasets
- [ ] Fix bugs, and bugs, and bugs ...
- [ ] Automatic ground estimation and Manhattan alignment

## Some notes

1. I made some changes to the original 3DGS. First of all, I took the hyperparameters of 3DGS from `arguments/__init__.py` and put them into `arguments/parameters.py` file to make it easier to read and understand the hyperparameters
2. In order not to change the original directory structure of 3DGS, I added a new `VastGaussian_scene` module to store VastGaussian. Part of the code I called the existing functions in the `scene` folder. Also to fix the `import` error, I moved the Scene class into the datasets.py folder

<div align="center">
    <img src=image/img2.png align="center"> 
    <img src=image/img_1.png align="center">
</div>

3. The naming of the files is consistent with the method mentioned in the paper for easy reading

> - `datasets.py` I have rewritten the Scene class in 3DGS into BigScene and PartitionScene. The former represents the original scene BigScene, and the latter represents the PartitionScene of each small scene after Partition.
> - `data_partition.py` corresponding to the `Progressive Data Partitioning` in the paper.
> 
> <img src=image/img_3.png align="center" width=800>
> 
> - `decouple_appearance_model.py`  corresponding to the `Decoupled Appearance Modeling` in the paper.
>  
>    <div align="center">
>        <img src=image/img.png align="center" height=400>
>        <img src=image/img_2.png align="center" width=400>
>    </div> 
>
> - `graham_scan.py` convex hull calculation is used to project the partition cube onto the camera plane and calculate the intersection of the projected region and the image region when implementing Visibility based camera selection.
> 
> - `seamless_merging.py` corresponding to the `Seamless Merging` in the paper.

4. I have added a new file `train_vast.py` to modify the process of training VastGaussian, if you want to train the original 3DGS, please use `train.py`.
5. The paper mentioned `Manhattan world alignment, so that the Y-axis of the world coordinate is perpendicular to the ground plane`, I asked the experts to know that this thing can be adjusted manually using `threejs`: https://threejs.org/editor/ or the software `cloudcompare`, after manually adjusting the scene you get the --position and --rotation parameters, just take them as command line arguments and train.

> ## 1. Using `threejs` for Manhattan alignment
> - After importing your initial point cloud via File-->Import you can check if your initial point cloud needs to be Manhattan aligned, if it looks like this
>  <div align="center">
>       <img src=image/img_7.png align="center" width=600>
>  </div>
> - Now you can adjust your point cloud so that the ground is perpendicular to the y-axis and the boundaries are as parallel as possible to the x- and z-axis with the help of the options on the left, but of course you can also use the editing area on the right to directly enter the corresponding values.
>    <div align="center">
>        <img src=image/img_8.png  height=400>
>        <img src=image/img_9.png  height=700>
>    </div> 
> - Then you can get the appropriate parameters in the right edit area.
> 
> ## 2. Using `cloudcompare` for Manhattan alignment
> - Open the cloudcompare software and import the sparse point cloud into the software.
> <div align="center">
>  <img src="image/img_6.png" width="800">
> </div>
>
> - Use the `cross section` tool in the toolbar to reduce the scope of the point cloud to only the areas you are interested in (for easier alignment), or you can leave them out.
> Then you can use the toolbar on the left of the <a style="color: red">red arrow</a> to adjust your viewing Angle (there are 6 viewing angles), and finally drag the arrow pointed by the <a style="color: green">green arrow</a> to adjust the area you are interested in.
> <div align="center">
>   <img src="image/img_10.png" width="800">
>   <img src="image/img_11.png" width="800">
> </div>
>
> - After you have adjusted the point cloud, you can export it as a new point cloud, noting that there is no transformation of coordinates involved. Then close the box on the right.
> Select the exported point cloud and deselect the initial point cloud.
> <div align="center">
>  <img src="image/img_12.png" width="800">
>  <img src="image/img_13.png" width="800">
> </div>
>
> - Use the `Translate/Rotate` tool on the toolbar to adjust the pose of the point cloud.
> Click `Rotation` to select the axis around which you want to rotate. If you want to adjust both the rotation matrix and the transfer vector, you can tick `Ty Ty Tz`
> Also select the toolbar on the left to adjust the viewing Angle.
> <div align="center">
>  <img src="image/img_14.png" width="800">
>  <img src="image/img_15.png" width="800">
> </div>
>
> - The Manhatton alignment mentioned in the paper can be realized by manually adjusting the pose of the point cloud so that the boundary frame x and z axis of the point cloud are parallel.
> And you can get the transformation matrix relative to the initial point cloud after this adjustment in the software console. Let's call it `A1`
> <div align="center">
>  <img src="image/img_16.png" width="800">
>  <img src="image/img_17.png" width="800">
>  <img src="image/img_18.png" width="800">
>  <img src="image/img_18.png" width="800">
>  <img src="image/img_19.png" width="800">
> </div>
>
> - If you only adjust once, then A1 is the final transformation matrix (`A=A1`), if you adjust the pose of the point cloud several times in A row, assuming 3 adjustments, and get the transformation matrix `A1 A2 A3`, then the final transformation matrix is `A= A3*A2*A1`
> <div align="center">
>  <img src="image/img_20.png" width="800">
> </div>
> 
> - Enter the resulting transformation matrix into the command line.

6. In the process of implementation, I used a small range of data provided by 3DGS for testing. Larger data can not run on the native computer, and a large range of data requires at least **32G video memory** according to the instructions of the paper.
7. In the implementation process, some operations in the paper, the author is not very clear about the details, so some implementation is based on my guess and understanding to complete, so my implementation may have some bugs, and some implementation may be a little stupid in the eyes of the expert, if you find problems in the use of the process, please contact me in time, progress together.

## Using
1. The data format is the same as 3DGS, and the training command is basically the same as 3DGS. I didn't make too many personalized changes, you can refer to the following command (see `arguments/parameters.py` for more parameters):
if you want to perform manhattan alignment:

Using `threejs` for Manhattan alignment

```python
python train_vast.py -s datasets/xxx --exp_name xxx --manhattan --pos xx xx xx --rot xx xx xx
```

Using `cloudcompare` for Manhattan alignment

```python
# The 9 elements of the rotation matrix should be filled in rot
python train_vast.py -s datasets/xxx --exp_name xxx --manhattan --pos xx xx xx --rot xx xx xx xx xx xx xx xx xx
```

otherwise:
```python
python train_vast.py -s datasets/xxx --exp_name test
```

## Datasets
1. `Urbanscene3D`: https://github.com/Linxius/UrbanScene3D

2. `Mill-19`: https://opendatalab.com/OpenDataLab/Mill_19/tree/main/raw

3. test data for this implementation: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip

# Contributors
Happily, we now have several contributors working on the project, and we welcome more contributors to join us to improve the project. Thank you all for your work.

<a href="https://github.com/VerseWei">
  <img src="https://avatars.githubusercontent.com/u/102359772?v=4" height="75" width="75"/>
</a>
