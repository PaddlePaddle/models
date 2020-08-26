# UCF101数据准备
UCF101数据的相关准备。主要包括数据下载，视频文件的提取frames，以及生成文件的路径list

---
## 1. 数据下载
UCF101数据的详细信息可以参考网站[UCF101](https://www.crcv.ucf.edu/data/UCF101.php)。 为了方便用户使用，我们提供了UCF101数据的annotations文件和videos文件的下载脚本。

### 下载annotations文件
首先，请确保在`./data/dataset/ucf101/`目录下，输入如下UCF101数据集的标注文件的命令。
```shell
bash download_annotations.sh
```

### 下载UCF101的视频文件
同样需要确保在`./data/dataset/ucf101/`目录下，输入下述命令下载视频文件
```shell
bash download_annotations.sh
```
下载完成后视频文件会存储在`./data/dataset/ucf101/videos/`文件夹下

---
## 2. 提取视频文件的frames
为了加速网络的训练过程，我们首先对视频文件（ucf101视频文件为avi格式）提取帧 (frames)。通过读取frames的方式替换原始的直接读取视频文件，能够极大的减小巡训练的时间开销。

直接输入如下命令，即可提取ucf101视频文件的frames
``` python
python extract_rawframes_opencv.py ./videos/ ./rawframes/ --level 2 --ext avi
```

---
## 3. 生成frames文件和视频文件的路径list
生成视频文件的路径list，输入如下命令

```python
 python build_file_list.py videos/ --level 2 --format videos --out_list_path ./ --shuffle
```
生成frames文件的路径list，输入如下命令：
```python
 python build_file_list.py rawframes/ --level 2 --format rawframes --out_list_path ./ --shuffle
```

**参数说明**

`videos/` 或者 `rawframes/` ： 表示视频或者frames文件的存储路径

`--level 2` ： 表示文件的存储结构

`--format`： 表示是针对视频还是frames生成路径list

`--out_list_path `： 表示生的路径list文件存储位置

`--shuffle`： 表示对路径list中的文件顺序进行shuffle


# 以上步骤完成后，文件组织形式如下所示

```
├── data
|   ├── dataset
|   │   ├── ucf101
|   │   │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
|   │   │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
|   │   │   ├── annotations
|   │   │   ├── videos
|   │   │   │   ├── ApplyEyeMakeup
|   │   │   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi
|  
|   │   │   │   ├── YoYo
|   │   │   │   │   ├── v_YoYo_g25_c05.avi
|   │   │   ├── rawframes
|   │   │   │   ├── ApplyEyeMakeup
|   │   │   │   │   ├── v_ApplyEyeMakeup_g01_c01
|   │   │   │   │   │   ├── img_00001.jpg
|   │   │   │   │   │   ├── img_00002.jpg
|   │   │   │   │   │   ├── ...
|   │   │   │   │   │   ├── flow_x_00001.jpg
|   │   │   │   │   │   ├── flow_x_00002.jpg
|   │   │   │   │   │   ├── ...
|   │   │   │   │   │   ├── flow_y_00001.jpg
|   │   │   │   │   │   ├── flow_y_00002.jpg
|   │   │   │   ├── ...
|   │   │   │   ├── YoYo
|   │   │   │   │   ├── v_YoYo_g01_c01
|   │   │   │   │   ├── ...
|   │   │   │   │   ├── v_YoYo_g25_c05

```
