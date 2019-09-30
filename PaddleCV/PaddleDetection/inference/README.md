# PaddleDetection C++预测部署方案

## 说明

本目录提供一个跨平台的图像检测模型的C++预测部署方案，用户通过一定的配置，加上少量的代码，即可把模型集成到自己的服务中，完成相应的图像检测任务。

主要设计的目标包括以下五点：
- 跨平台，支持在 windows 和 Linux 完成编译、开发和部署
- 支持多种图像检测任务，用户通过少量配置即可加载模型完成常见检测任务，比如iphone划痕检测等
- 可扩展性，支持用户针对新模型开发自己特殊的数据预处理、后处理等逻辑
- 高性能，除了`PaddlePaddle`自身带来的性能优势，我们还针对图像检测的特点对关键步骤进行了性能优化
- 支持常见的图像检测模型，yolov3、faster rcnn以及faster rcnn+fpn
 

## 主要目录和文件

```bash
deploy
├── detection_demo.cpp # 完成图像检测预测任务C++代码
│
├── conf
│   ├── detection_yolov3.yaml # 示例yolov3目标检测配置
│   ├── detection_rcnn.yaml #示例faster rcnn 目标检测配置
│   └── detection_rcnn_fpn.yaml #示例faster rcnn + fpn目标检测配置
├── images
│   ├── detection # 示例yolov3目标检测测试图片目录
│   └── detection_rcnn # 示例faster rcnn + fpn目标检测测试图片目录
├── tools
│   └── visualize.py # 示例人像分割模型结果可视化脚本
├── docs
│   ├── linux_build.md # Linux 编译指南
│   ├── windows_vs2015_build.md # windows VS2015编译指南
│   └── windows_vs2019_build.md # Windows VS2019编译指南
│
├── utils # 一些基础公共函数
│
├── preprocess # 数据预处理相关代码
│
├── predictor # 模型加载和预测相关代码
│
├── CMakeList.txt # cmake编译入口文件
│
└── external-cmake # 依赖的外部项目cmake（目前仅有yaml-cpp）

```

## 编译
支持在`Windows`和`Linux`平台编译和使用：
- [Linux 编译指南](./docs/linux_build.md)
- [Windows 使用 Visual Studio 2019 Community 编译指南](./docs/windows_vs2019_build.md)
- [Windows 使用 Visual Studio 2015 编译指南](./docs/windows_vs2015_build.md)

`Windows`上推荐使用最新的`Visual Studio 2019 Community`直接编译`CMake`项目。

## 预测并可视化结果

完成编译后，便生成了需要的可执行文件和链接库。这里以部署iphone划痕检测模型为例，介绍搭建图像检测模型的通用流程。

### 1. 下载模型文件
我们提供了一个iphone划痕检测模型示例用于测试，点击右侧地址下载：[yolov3示例模型下载地址](https://paddleseg.bj.bcebos.com/inference/yolov3_darknet_iphone.zip)。（还提供faster rcnn，faster rcnn+fpn示例模型，可在以下链接下载：[faster rcnn示例模型下载地址](https://paddleseg.bj.bcebos.com/inference/faster_rcnn_pp50.zip)，
 [faster rcnn + fpn示例模型下载地址](https://paddleseg.bj.bcebos.com/inference/faster_rcnn_pp50_fpn.zip)）

下载并解压，解压后目录结构如下：
```
yolov3_darknet_iphone
├── __model__ # 模型文件
│
└── __params__ # 参数文件
```
解压后把上述目录拷贝到合适的路径：

**假设**`Windows`系统上，我们模型和参数文件所在路径为`D:\projects\models\yolov3_darknet_iphone`。

**假设**`Linux`上对应的路径则为`/root/projects/models/yolov3_darknet_iphone`。


### 2. 修改配置

`inference`源代码(即本目录)的`conf`目录下提供了示例iphone划痕检测模型的配置文件`detection_yolov3.yaml`, 相关的字段含义和说明如下：

```yaml
DEPLOY:
    # 是否使用GPU预测
    USE_GPU: 1
    # 模型和参数文件所在目录路径
    MODEL_PATH: "/root/projects/models/yolov3_darknet_iphone"
    # 模型文件名
    MODEL_FILENAME: "__model__"
    # 参数文件名
    PARAMS_FILENAME: "__params__"
    # 预测图片的标准输入，尺寸不一致会resize
    EVAL_CROP_SIZE: (608, 608)
    # resize方式，支持 UNPADDING和RANGE_SCALING
    RESIZE_TYPE: "UNPADDING"
    # 短边对齐的长度，仅在RANGE_SCALING下有效
    TARGET_SHORT_SIZE : 256
    # 均值
    MEAN:  [0.4647, 0.4647, 0.4647]
    # 方差
    STD: [0.0834, 0.0834, 0.0834]
    # 图片类型， rgb或者rgba
    IMAGE_TYPE: "rgb"
    # 像素分类数
    NUM_CLASSES: 1
    # 通道数
    CHANNELS : 3
    # 预处理器， 目前提供图像检测的通用处理类SegPreProcessor
    PRE_PROCESSOR: "DetectionPreProcessor"
    # 预测模式，支持 NATIVE 和 ANALYSIS
    PREDICTOR_MODE: "ANALYSIS"
    # 每次预测的 batch_size
    BATCH_SIZE : 3 
    # 长边伸缩的最大长度，-1代表无限制。
    RESIZE_MAX_SIZE: -1
    # resize后的需要裁剪的大小。
    CROP_SIZE: (608, 608)
    # 输入的tensor数量。
    FEEDS_SIZE: 2

```
修改字段`MODEL_PATH`的值为你在**上一步**下载并解压的模型文件所放置的目录即可。更多配置文件字段介绍，请参考文档[预测部署方案配置文件说明](./docs/configuration.md)。

### 3. 执行预测

在终端中切换到生成的可执行文件所在目录为当前目录(Windows系统为`cmd`)。

`Linux` 系统中执行以下命令：
```shell
./detection_demo --conf=conf/detection_yolov3.yaml --input_dir=images/detection
```
`Windows` 中执行以下命令:
```shell
.\detection_demo.exe --conf=conf\\detection_yolov3.yaml --input_dir=images\detection\\
```


预测使用的两个命令参数说明如下：

| 参数 | 含义 |
|-------|----------|
| conf | 模型配置的Yaml文件路径 |
| input_dir | 需要预测的图片目录 |


配置文件说明请参考上一步，样例程序会扫描input_dir目录下的所有图片，并生成对应的预测结果到屏幕，并保存到X.pb文件（X为对应图片的文件名）. 可使用工具脚本detection_visualization.py将检测结果可视化。

```bash 
python detection_visualization.py img img.pb
```

检测结果（每个图片的结果用空行隔开）

```原图：```
![原图](./demo_images/00000001.jpg)

```检测结果图：```
![检测结果](./demo_images/00000001.jpg.png)