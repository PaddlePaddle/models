# PaddlePaddle Models

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/PaddlePaddle/models)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

PaddlePaddle provides a rich set of computational units to enable users to adopt a modular approach to solving various learning problems. In this repo, we demonstrate how to use PaddlePaddle to solve common machine learning tasks, providing several different neural network model that anyone can easily learn and use.


- [fluid models](fluid): use PaddlePaddle's Fluid APIs. We especially recommend users to use Fluid models.


PaddlePaddle 提供了丰富的计算单元，使得用户可以采用模块化的方法解决各种学习问题。在此repo中，我们展示了如何用 PaddlePaddle 来解决常见的机器学习任务，提供若干种不同的易学易用的神经网络模型。

- [fluid模型](fluid): 使用 PaddlePaddle Fluid版本的 APIs，我们特别推荐您使用Fluid模型。

## PaddleCV
模型|简介|模型优势|参考论文
--|:--:|:--:|:--:
[AlexNet](./PaddleCV/image_classification/models)|图像分类经典模型|首次在CNN中成功的应用了ReLU、Dropout和LRN，并使用GPU进行运算加速|[ImageNet Classification with Deep Convolutional Neural Networks](https://www.researchgate.net/publication/267960550_ImageNet_Classification_with_Deep_Convolutional_Neural_Networks)
[VGG](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models)|图像分类经典模型|在AlexNet的基础上使用3*3小卷积核，增加网络深度，具有很好的泛化能力|[Very Deep ConvNets for Large-Scale Inage Recognition](https://arxiv.org/pdf/1409.1556.pdf)
[GoogleNet](./PaddleCV/image_classification/models)|图像分类经典模型|在不增加计算负载的前提下增加了网络的深度和宽度，性能更加优越|[Going deeper with convolutions](https://ieeexplore.ieee.org/document/7298594)
[ResNet](./PaddleCV/image_classification/models)|残差网络|引入了新的残差结构，解决了随着网络加深，准确率下降的问题|[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
[Inception-v4](./PaddleCV/image_classification/models)|图像分类经典模型|更加deeper和wider的inception结构|[Inception-ResNet and the Impact of Residual Connections on Learning](http://arxiv.org/abs/1602.07261)
[MobileNet](./PaddleCV/image_classification/models)|轻量级网络模型|为移动和嵌入式设备提出的高效模型|[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
[DPN](./PaddleCV/image_classification/models)|图像分类模型|结合了DenseNet和ResNeXt的网络结构，对图像分类效果有所提升|[Dual Path Networks](https://arxiv.org/abs/1707.01629)
[SE-ResNeXt](./PaddleCV/image_classification/models)|图像分类模型|ResNeXt中加入了SE block，提高了模型准确率|[Squeeze-and-excitation networks](https://arxiv.org/abs/1709.01507)
[SSD](./PaddleCV/object_detection/README_cn.md)|单阶段目标检测器|在不同尺度的特征图上检测对应尺度的目标,可以方便地插入到任何一种标准卷积网络中|[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
[YOLOv3](./PaddleCV/yolov3/README_cn.md)|单阶段目标检测器|基于darknet53主干网络在多种尺度的特征图上进行端到端实时目标检测,检测速度快|[YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
[Face Detector: PyramidBox](./PaddleCV/face_detection/README_cn.md)|基于SSD的单阶段人脸检测器|利用上下文信息解决困难人脸的检测问题，网络表达能力高，鲁棒性强|[PyramidBox: A Context-assisted Single Shot Face Detector](https://arxiv.org/pdf/1803.07737.pdf)
[Faster RCNN](./PaddleCV/rcnn/README_cn.md)|典型的两阶段目标检测器|创造性地采用卷积网络自行产生建议框，并且和目标检测网络共享卷积网络，建议框数目减少，质量提高|[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
[Mask RCNN](./PaddleCV/rcnn/README_cn.md)|基于Faster RCNN模型的经典实例分割模型|在原有Faster RCNN模型基础上添加分割分支，得到掩码结果，实现了掩码和类别预测关系的解藕。|[Mask R-CNN](https://arxiv.org/abs/1703.06870)
[ICNet](./PaddleCV/icnet)|图像实时语义分割模型|即考虑了速度，也考虑了准确性，在高分辨率图像的准确性和低复杂度网络的效率之间获得平衡|[ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545)
[DCGAN](./PaddleCV/gan/c_gan)|图像生成模型|深度卷积生成对抗网络，将GAN和卷积网络结合起来，以解决GAN训练不稳定的问题|[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)
[ConditionalGAN](./PaddleCV/gan/c_gan)|图像生成模型|条件生成对抗网络，一种带条件约束的GAN，使用额外信息对模型增加条件，可以指导数据生成过程|[Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
[CycleGAN](./PaddleCV/gan/cycle_gan)|图片转化模型|自动将某一类图片转换成另外一类图片，可用于风格迁移|[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
[CRNN-CTC模型](./PaddleCV/ocr_recognition)|场景文字识别模型|使用CTC model识别图片中单行英文字符|[Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.researchgate.net/publication/221346365_Connectionist_temporal_classification_Labelling_unsegmented_sequence_data_with_recurrent_neural_'networks)
[Attention模型](./PaddleCV/ocr_recognition)|场景文字识别模型|使用attention 识别图片中单行英文字符|[Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247)
[Metric Learning](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning)|度量学习模型|能够用于分析对象时间的关联、比较关系，可应用于辅助分类、聚类问题，也广泛用于图像检索、人脸识别等领域|-
[TSN](./PaddleCV/video_classification)|视频分类模型|基于长范围时间结构建模，结合了稀疏时间采样策略和视频级监督来保证使用整段视频时学习得有效和高效|[Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859)
[视频模型库](./PaddleCV/video)|视频模型库|给开发者提供基于PaddlePaddle的便捷、高效的使用深度学习算法解决视频理解、视频编辑、视频生成等一系列模型||
[caffe2fluid](./PaddleCV/caffe2fluid)|将Caffe模型转换为Paddle Fluid配置和模型文件工具|-|-

## PaddleNLP
模型|简介|模型优势|参考论文
--|:--:|:--:|:--:
[Transformer](./PaddleNLP/neural_machine_translation/transformer/README.md)|机器翻译模型|基于self-attention，计算复杂度小，并行度高，容易学习长程依赖，翻译效果更好|[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
[BERT](https://github.com/PaddlePaddle/LARK/tree/develop/BERT)|语义表示模型|在多个 NLP 任务上取得 SOTA 效果，支持多卡多机训练，支持混合精度训练|[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
[ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)|语义表示模型|基于知识增强的中文语义表示模型，在多个任务上的效果超越 BERT 中文模型|-
[ELMo](https://github.com/PaddlePaddle/LARK/tree/develop/ELMo)|语义表示模型|支持多卡训练，训练速度比主流实现快1倍，提供在中文词法分析任务上迁移学习的示例。|[ELMo: Embeddings from Language Models](https://arxiv.org/abs/1802.05365)
[LAC](https://github.com/baidu/lac/blob/master/README.md)|联合的词法分析模型|能够整体性地完成中文分词、词性标注、专名识别任务|[Chinese Lexical Analysis with Deep Bi-GRU-CRF Network](https://arxiv.org/abs/1807.01882)
[Senta](https://github.com/baidu/Senta/blob/master/README.md)|情感倾向分析模型集|百度AI开放平台中情感倾向分析模型|-
[DAM](./PaddleNLP/dialogue_model_toolkit/deep_attention_matching)|语义匹配模型|百度自然语言处理部发表于ACL-2018的工作,用于检索式聊天机器人多轮对话中应答的选择|[Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network](http://aclweb.org/anthology/P18-1103)
[SimNet](https://github.com/baidu/AnyQ/blob/master/tools/simnet/train/paddle/README.md)|语义匹配框架|使用SimNet构建出的模型可以便捷的加入AnyQ系统中，增强AnyQ系统的语义匹配能力|-
[DuReader](./PaddleNLP/reading_comprehension/README.md)|阅读理解模型|百度MRC数据集上的机器阅读理解模型|-
[dialogue model](https://github.com/baidu/knowledge-driven-dialogue/tree/master/generative_paddle/README.md)|知识驱动的对话模型|基于双向RNN和attention实现的生成式对话系统|-


## PaddleRec
模型|简介|模型优势|参考论文
--|:--:|:--:|:--:
[TagSpace](./PaddleRec/tagspace)|文本及标签的embedding表示学习模型|应用于工业级的标签推荐，具体应用场景有feed新闻标签推荐等|[#TagSpace: Semantic embeddings from hashtags](https://www.bibsonomy.org/bibtex/0ed4314916f8e7c90d066db45c293462)
[GRU4Rec](./PaddleRec/gru4rec)|个性化推荐模型|首次将RNN（GRU）运用于session-based推荐，相比传统的KNN和矩阵分解，效果有明显的提升|[Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)
[SSR](./PaddleRec/ssr)|序列语义检索推荐模型|使用参考论文中的思想，使用多种时间粒度进行用户行为预测|[Multi-Rate Deep Learning for Temporal Recommendation](https://dl.acm.org/citation.cfm?id=2914726)
[DeepCTR](./PaddleRec/ctr/README.cn.md)|点击率预估模型|只实现了DeepFM论文中介绍的模型的DNN部分，DeepFM会在其他例子中给出|[DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)
[Multiview-Simnet](./PaddleRec/multiview_simnet)|个性化推荐模型|基于多元视图，将用户和项目的多个功能视图合并为一个统一模型|[A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems](http://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1159-songA.pdf)

## Other Models
模型|简介|模型优势|参考论文
--|:--:|:--:|:--:
[DeepASR](./PaddleSpeech/DeepASR/README_cn.md)|语音识别系统|利用Fluid框架完成语音识别中声学模型的配置和训练，并集成 Kaldi 的解码器|-
[DQN](./PaddleRL/DeepQNetwork/README_cn.md)|深度Q网络|value based强化学习算法，第一个成功地将深度学习和强化学习结合起来的模型|[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
[DoubleDQN](./PaddleRL/DeepQNetwork/README_cn.md)|DQN的变体|将Double Q的想法应用在DQN上，解决过优化问题|[Font Size: Deep Reinforcement Learning with Double Q-Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12389)
[DuelingDQN](./PaddleRL/DeepQNetwork/README_cn.md)|DQN的变体|改进了DQN模型，提高了模型的性能|[Dueling Network Architectures for Deep Reinforcement Learning](http://proceedings.mlr.press/v48/wangf16.html)

## License
This tutorial is contributed by [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) and licensed under the [Apache-2.0 license](LICENSE).


## 许可证书
此向导由[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)贡献，受[Apache-2.0 license](LICENSE)许可认证.
