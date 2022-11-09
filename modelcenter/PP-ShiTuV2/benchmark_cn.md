## 1. 训练Benchmark

### 1.1 软硬件环境

* PP-ShiTuV2 的特征提取模型训练过程中使用8 GPUs，每GPU batch size为256进行训练，采样器使用PKSampler，一个含256个样本mini-batch有64个类别，每个类别内含4张不同的图片，如训练GPU数和batch size不使用上述配置，须参考FAQ调整学习率和迭代次数。

  **注**：由于本模型使用PKSampler和metric learning相关方法，因此改变batch size可能对性能有比较明显的影响。

* PP-ShiTuV2 的检测模型训练过程中使用8 GPUs，每GPU batch size为28进行训练，如训练GPU数和batch size不使用上述配置，须参考FAQ调整学习率和迭代次数。

### 1.2 数据集
特征提取模型对原有的训练数据进行了合理扩充与优化，最终使用如下 17 个公开数据集的汇总：

| 数据集                 | 数据量  |  类别数  | 场景  |                                      数据集地址                                      |
| :--------------------- | :-----: | :------: | :---: | :----------------------------------------------------------------------------------: |
| Aliproduct             | 2498771 |  50030   | 商品  |      [地址](https://retailvisionworkshop.github.io/recognition_challenge_2020/)      |
| GLDv2                  | 1580470 |  81313   | 地标  |               [地址](https://github.com/cvdfoundation/google-landmark)               |
| VeRI-Wild              | 277797  |  30671   | 车辆  |                    [地址](https://github.com/PKU-IMRE/VERI-Wild)                     |
| LogoDet-3K             | 155427  |   3000   | Logo  |              [地址](https://github.com/Wangjing1551/LogoDet-3K-Dataset)              |
| SOP                    |  59551  |  11318   | 商品  |              [地址](https://cvgl.stanford.edu/projects/lifted_struct/)               |
| Inshop                 |  25882  |   3997   | 商品  |            [地址](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)             |
| bird400                |  58388  |   400    | 鸟类  |          [地址](https://www.kaggle.com/datasets/gpiosenka/100-bird-species)          |
| 104flows               |  12753  |   104    | 花类  |              [地址](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)              |
| Cars                   |  58315  |   112    | 车辆  |            [地址](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)            |
| Fashion Product Images |  44441  |    47    | 商品  | [地址](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) |
| flowerrecognition      |  24123  |    59    | 花类  |         [地址](https://www.kaggle.com/datasets/aymenktari/flowerrecognition)         |
| food-101               | 101000  |   101    | 食物  |         [地址](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)          |
| fruits-262             | 225639  |   262    | 水果  |            [地址](https://www.kaggle.com/datasets/aelchimminut/fruits262)            |
| inaturalist            | 265213  |   1010   | 自然  |           [地址](https://github.com/visipedia/inat_comp/tree/master/2017)            |
| indoor-scenes          |  15588  |    67    | 室内  |       [地址](https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019)       |
| Products-10k           | 141931  |   9691   | 商品  |                       [地址](https://products-10k.github.io/)                        |
| CompCars               |  16016  |   431    | 车辆  |     [地址](http://​​​​​​http://ai.stanford.edu/~jkrause/cars/car_dataset.html​)      |
| **Total**              | **6M**  | **192K** |   -   |                                          -                                           |


主体检测模型的数据集请参考 [主体检测模型数据集](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.5/docs/zh_CN/training/PP-ShiTu/mainbody_detection.md#1-%E6%95%B0%E6%8D%AE%E9%9B%86)

### 1.3 指标 （字段可根据模型情况，自行定义）

| 模型名称                                 | 模型简介 | 模型体积     | 输入尺寸 | ips |
| ---------------------------------------- | -------- | ------------ | -------- | --- |
| picodet_lcnet_x2_5_640_mainbody.yml      | 主体检测 | 30MB(KL量化) | 640      | 21 |
| GeneralRecognitionV2_PPLCNetV2_base.yaml | 特征提取 | 19MB(KL量化) | 224      | 163 |


## 2. 推理 Benchmark

### 2.1 软硬件环境

* PP-ShiTuV2主体检测和特征提取模型的推理速度测试采用CPU，开启MKLDNN，10线程，batch size=1进行测试。


### 2.2 数据集

PP-ShiTuV2特征提取模型使用自建产品数据集作为测试集

### 2.3 指标（字段可根据模型情况，自行定义）

| 模型       | 存储(主体检测+特征提取) | product |
| :--------- | :---------------------- | :------------------ |
|            |                         | recall@1            |
| PP-ShiTuV1 | 64(30+34)MB             | 66.8%                 |
| PP-ShiTuV2 | 49(30+19)MB               | 73.8%                 |


## 3. 相关使用说明
请参考：https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PP-ShiTu/README.md
