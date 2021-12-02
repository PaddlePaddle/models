# 论文复现赛指南-推荐方向

> 本文为针对 `推荐` 方向的复现赛指南
> 
> 如果希望查阅 `CV` 方向的复现赛指南，可以参考：[CV方向论文复现赛指南](./ArticleReproduction_CV.md)
> 
> 如果希望查阅 `NLP` 方向的复现赛指南，可以参考：[NLP方向论文复现赛指南](./ArticleReproduction_NLP.md)

## 目录

- [1. 总览](#1)
    - [1.1 背景](#1.1)
    - [1.2 前序工作](#1.2)
- [2. 整体框图](#2)
- [3. 论文复现理论知识](#3)
- [4. 论文复现注意事项与FAQ](#4)
    - [4.1 通用注意事项](#4.0)
    - [4.2 TIPC基础链条测试接入](#4.1)
- [5. 合入代码规范要求](#5)

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
* 除了精度符合要求之外，还需要符合代码规范（详见第五章）。
* 飞桨训推一体认证 (Training and Inference Pipeline Certification, TIPC) 是一个针对飞桨模型的测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。论文训练对齐之后，需要为代码接入TIPC基础链条测试文档与代码，关于TIPC基础链条测试接入规范的文档可以参考：[链接](https://github.com/PaddlePaddle/models/blob/tipc/docs/tipc_test/development_specification_docs/train_infer_python.md)。更多内容在`4.2`章节部分也会详细说明。


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
    
<a name="4.1"></a>
### 4.2 TIPC基础链条接入

**【基本流程】**

* 完成模型的训练、导出inference、基于PaddleInference的推理过程的文档与代码。参考链接：
    * [insightface训练预测使用文档](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_paddle/README_cn.md)
    * [PaddleInference使用文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/inference_cn.html)
    * [PaddleRecInference使用文档](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/inference.md)
* 基于[TIPC基础链条测试接入规范](https://github.com/PaddlePaddle/models/blob/tipc/docs/tipc_test/development_specification_docs/train_infer_python.md)，完成该模型的TIPC基础链条开发以及测试文档/脚本，目录为`test_tipc`，测试脚本名称为`test_train_inference_python.sh`，该任务中只需要完成`少量数据训练模型，少量数据预测`的模式即可，用于测试TIPC流程的模型和少量数据需要放在当前repo中。


**【注意事项】**

* 基础链条测试接入时，只需要验证`少量数据训练模型，少量数据预测`的模式，只需要在Linux下验证通过即可。
* 在文档中需要给出一键测试的脚本与使用说明。
* 禁止修改通用参数, 比如train_infer_python.txt中的enable_tensorRT，enable_mkldnn等。 
* 接入TIPC功能是需安装[特定版本paddle](https://paddleinference.paddlepaddle.org.cn/master/user_guides/download_lib.html#python)。

**【实战】**

TIPC基础链条测试接入用例可以参考：[PaddlRec TIPC基础链条测试开发文档](https://github.com/PaddlePaddle/PaddleRec/tree/master/test_tipc), [InsightFace-paddle TIPC基础链条测试开发文档](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_paddle/test_tipc/readme.md)。


**【验收】**

* TIPC基础链条测试文档清晰，`test_train_inference_python.sh`脚本可以成功执行并返回正确结果。

<a name="5"></a>
## 5. 合入代码规范要求

验收的最后一道标准是需要符合合入官方套件的要求，具体要求见[链接](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/contribute.md)


