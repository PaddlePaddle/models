## 1. 训练Benchmark

### 1.1 软硬件环境

* PP-ShiTu 的特征提取模型训练过程中使用8 GPUs，每GPU batch size为256进行训练，如训练GPU数和batch size不使用上述配置，须参考FAQ调整学习率和迭代次数。

* PP-ShiTu 的检测模型训练过程中使用8 GPUs，每GPU batch size为28进行训练，如训练GPU数和batch size不使用上述配置，须参考FAQ调整学习率和迭代次数。

### 1.2 数据集
特征提取模型对原有的训练数据进行了合理扩充与优化，最终使用如下 17 个公开数据集的汇总：

|    数据集    | 数据量  |  类别数  |   场景   |                                  数据集地址                                  |
| :----------: | :-----: | :------: | :------: | :--------------------------------------------------------------------------: |
|  Aliproduct  | 2498771 |  50030   |   商品   |  [地址](https://retailvisionworkshop.github.io/recognition_challenge_2020/)  |
|    GLDv2     | 1580470 |  81313   |   地标   |           [地址](https://github.com/cvdfoundation/google-landmark)           |
|  VeRI-Wild   | 277797  |  30671   |   车辆   |                [地址](https://github.com/PKU-IMRE/VERI-Wild)                 |
|  LogoDet-3K  | 155427  |   3000   |   Logo   |          [地址](https://github.com/Wangjing1551/LogoDet-3K-Dataset)          |
| iCartoonFace | 389678  |   5013   | 动漫人物 | [地址](http://challenge.ai.iqiyi.com/detail?raceId=5def69ace9fcf68aef76a75d) |
|     SOP      |  59551  |  11318   |   商品   |          [地址](https://cvgl.stanford.edu/projects/lifted_struct/)           |
|    Inshop    |  25882  |   3997   |   商品   |        [地址](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)         |
|  **Total**   | **5M**  | **185K** |   ----   |                                     ----                                     |

主体检测模型的数据集请参考 [主体检测模型数据集](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/image_recognition_pipeline/mainbody_detection.md#1-%E6%95%B0%E6%8D%AE%E9%9B%86)

### 1.3 指标 （字段可根据模型情况，自行定义）

| 模型名称                             | 模型简介 | 模型体积   | 输入尺寸 | ips |
| ------------------------------------ | -------- | ---------- | -------- | --- |
| picodet_lcnet_x2_5_640_mainbody.yml  | 主体检测 | 30MB(量化) | 640      | 21  |
| GeneralRecognition_PPLCNet_x2_5.yaml | 特征提取 | 34MB(量化) | 224      | 200 |


## 2. 推理 Benchmark

### 2.1 软硬件环境

* PP-ShiTu主体检测和特征提取模型的推理速度测试采用CPU，开启MKLDNN，10线程，batch size=1进行测试。


### 2.2 数据集

PP-ShiTu特征提取模型使用自建产品数据集作为测试集

### 2.3 指标（字段可根据模型情况，自行定义）

| 模型     | 存储(主体检测+特征提取) | product  |
| :------- | :---------------------- | :------- |
|          |                         | recall@1 |
| PP-ShiTu | 64(30+34)MB             | 66.8%    |


## 3. 相关使用说明
请参考：https://github.com/PaddlePaddle/PaddleClas/tree/release/2.4/docs/zh_CN/image_recognition_pipeline
