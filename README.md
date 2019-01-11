# PaddlePaddle Models

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/PaddlePaddle/models)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

PaddlePaddle provides a rich set of computational units to enable users to adopt a modular approach to solving various learning problems. In this repo, we demonstrate how to use PaddlePaddle to solve common machine learning tasks, providing several different neural network model that anyone can easily learn and use.


- [fluid models](fluid): use PaddlePaddle's Fluid APIs. We especially recommend users to use Fluid models.


PaddlePaddle 提供了丰富的计算单元，使得用户可以采用模块化的方法解决各种学习问题。在此repo中，我们展示了如何用 PaddlePaddle 来解决常见的机器学习任务，提供若干种不同的易学易用的神经网络模型。

- [fluid模型](fluid): 使用 PaddlePaddle Fluid版本的 APIs，我们特别推荐您使用Fluid模型。

## PaddleCV
-  [AlexNet](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models)：图像分类经典模型。
-  [VGG](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models)：图像分类经典模型。
-  [GoogleNet](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models)：图像分类经典模型。
-  [Residual Network](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models)：残差网络。
-  [Inception-v4](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models)：Inception系列v4版本。
-  [MobileNet](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models)：轻量化网络模型。
-  [Dual Path Network](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models)：结合了DenseNet和ResNet的网络结构。
-  [SE-ResNeXt](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models)：ResNeXt中加入SE block。
-  [Caffe模型转换为Paddle Fluid配置和模型文件工具](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/caffe2fluid)：将Caffe模型转化为PaddlePaddle的工具。
-  [Single Shot MultiBox Detector](https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleCV/object_detection/README_cn.md)：单阶段目标检测器。
-  [Face Detector: PyramidBox](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/face_detection/README_cn.md)：基于SSD的单阶段人脸检测器
-  [Faster RCNN](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/faster_rcnn/README_cn.md)：典型的两阶段目标检测器。
-  [ICNet](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/icnet)：图像实时语义分割网络。
- [DCGAN & ConditionalGAN](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/gan/c_gan)：深度卷积生成对抗网络&条件深度卷积生成对抗网络。
- [CycleGAN](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/gan/cycle_gan)：un-paired图像转化模型。
-  [CRNN-CTC模型](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/ocr_recognition)：使用CTC model识别图片中单行英文字符
-  [Attention模型](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/ocr_recognition)：使用attention model识别图片中单行英文字符。
- [Metric Learning](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/metric_learning)：深度度量学习模型。
- [TSN](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/video_classification)：基于Temporal Segment Network的视频分类。

## PaddleNLP
-  [Transformer](https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleNLP/neural_machine_translation/transformer/README_cn.md)：使用Attention机制实现Seq2Seq建模的机器翻译模型。
- [LAC](https://github.com/baidu/lac/blob/master/README.md)：联合的词法分析模型。
- [Senta](https://github.com/baidu/Senta/blob/master/README.md)：情感倾向分析模型集。
- [Deep Attention Matching Network](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleNLP/deep_attention_matching_net)：语义匹配模型，用于检索式聊天机器人多轮对话中应答的选择。
-  [SimNet](https://github.com/baidu/AnyQ/blob/master/tools/simnet/train/paddle/README.md)：百度自然语言处理部自主研发的语义匹配框架。
-  [DuReader](https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleNLP/machine_reading_comprehension/README.md)：百度MRC数据集上的机器阅读理解模型。
-  [Bi-GRU-CRF](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleNLP/sequence_tagging_for_ner/README.md):命名实体识别。

## PaddleRec
- [TagSpace](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleRec/tagspace)：文本及标签的embedding表示学习模型。
- [GRU4Rec](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleRec/gru4rec)：应用了GRU的session-based推荐模型。
- [Sequence Semantic Retrieval](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleRec/ssr)：序列语义检索推荐模型
- [DeepCTR](https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleRec/ctr/README.cn.md)：基于DNN模型的点击率预估模型。
- [Multiview-Simnet](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleRec/multiview_simnet)：多元视图的个性化推荐模型。

## License
This tutorial is contributed by [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) and licensed under the [Apache-2.0 license](LICENSE).


## 许可证书
此向导由[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)贡献，受[Apache-2.0 license](LICENSE)许可认证.
