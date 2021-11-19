# 论文复现指南

## 目录

- [1. 总览](#1)
    - [1.1 背景](#1.1)
    - [1.2 前序工作](#1.2)
- [2. 整体框图](#2)
- [3. 论文复现理论知识](#3)
- [4. 论文复现注意事项与FAQ](#4)
    - [4.1 通用注意事项](#4.0)

<a name="1"></a>
## 1. 总览

<a name="1.1"></a>
### 1.1 背景

* 以深度学习为核心的人工智能技术仍在高速发展，通过论文复现，开发者可以获得
    * 学习成长：自我能力提升
    * 技术积累：对科研或工作有所帮助和启发
    * 社区荣誉：成果被开发者广泛使用

<a name="1.2"></a>
### 1.2 前序工作

基于本指南复现论文过程中，建议开发者准备以下内容。

* 熟悉paddle
    * 文档和API
        * 80%以上的API在功能上与pytorch相同
        * [PaddlePaddle文档链接](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)
        * [Pytorch-Paddlepaddle API映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/08_api_mapping/pytorch_api_mapping_cn.html)
    * [10分钟快速上手飞浆](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/01_quick_start_cn.html)
    * 数据处理[DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/DataLoader_cn.html)
* 了解该数据格式。以Criteo数据集为例，该数据包含了13个连续特征和26个离散特征；还有一个标签，点击用1表示，未点击用0表示。
* 准备好训练/验证数据集，用于模型训练与评估
* 准备好fake input data以及label，与模型输入shape、type等保持一致，用于后续模型前向对齐。
    * 在对齐模型前向过程中，我们不需要考虑数据集模块等其他模块，此时使用fake data是将模型结构和数据部分解耦非常合适的一种方式。
    * 将fake data以文件的形式存储下来，也可以保证PaddlePaddle与参考代码的模型结构输入是完全一致的，更便于排查问题。
    * 在该步骤中，以AlexNet为例，生成fake data的脚本可以参考：[gen_fake_data.py](https://github.com/littletomatodonkey/AlexNet-Prod/blob/master/pipeline/fake_data/gen_fake_data.py)。
* 在特定设备(CPU/GPU)上，跑通参考代码的预测过程(前向)以及至少2轮(iteration)迭代过程，保证后续基于PaddlePaddle复现论文过程中可对比。
* 在复现的过程中，只需要将PaddlePaddle的复现代码以及打卡日志上传至github，不能在其中添加参考代码的实现，在验收通过之后，需要删除打卡日志。建议在初期复现的时候，就将复现代码与参考代码分成2个文件夹进行管理。

<a name="2"></a>
## 2. 整体框图
可参考[cv部分](https://github.com/PaddlePaddle/models/blob/tipc/docs/lwfx/ThesisReproduction_CV.md)

<a name="3"></a>
## 3. 论文复现理论知识
可参考[cv部分](https://github.com/PaddlePaddle/models/blob/tipc/docs/lwfx/ThesisReproduction_CV.md)


<a name="4"></a>
## 4. 论文复现注意事项与FAQ

本部分主要总结大家在论文复现赛过程中遇到的问题，如果本章内容没有能够解决你的问题，欢迎给该文档提出优化建议或者给Paddle提[ISSUE](https://github.com/PaddlePaddle/Paddle/issues/new/choose)。

<a name="4.0"></a>
### 4.1 通用注意事项

* 常见问题和误区
    * 不要主动调参，目的是复现而不是提升精度
    * 不要加论文中没提到的模型结构
    * 数据和指标先行对齐
* 数据集获取
    * PaddleRec提供了大量推荐数据集，可优先从[这里查找](https://github.com/PaddlePaddle/PaddleRec/tree/master/datasets)
