# Research

发布基于飞桨的前沿研究工作，包括CV、NLP、KG、STDM等领域的顶会论文和比赛冠军模型。

## 目录
* [计算机视觉(Computer Vision)](#计算机视觉)
* [自然语言处理(Natrual Language Processing)](#自然语言处理)
* [知识图谱(Knowledge Graph)](#知识图谱)
* [时空数据挖掘(Spatial-Temporal Data-Mining)](#时空数据挖掘)

## 计算机视觉
| 任务类型     | 目录                                                         | 简介                                                         | 论文链接 |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| 图像检索 | [GNN-Re-Ranking](https://github.com/PaddlePaddle/Research/CV/GNN-Re-Ranking/) | 基于GNN的快速图像检索Re-Ranking。 | https://arxiv.org/abs/2012.07620v2 |
| 车流统计 | [VehicleCounting](https://github.com/PaddlePaddle/Research/CV/VehicleCounting/) | AICITY2020 车流统计竞赛datasetA TOP1 方案。 | - |
| 车辆再识别 | [PaddleReid](https://github.com/PaddlePaddle/Research/CV/PaddleReid) | 给定目标车辆，在检索库中检索同id车辆，支持多种特征子网络。 | - |
| 车辆异常检测 | [AICity2020-Anomaly-Detection](https://github.com/PaddlePaddle/Research/CV/AICity2020-Anomaly-Detection) |  在监控视频中检测车辆异常情况，例如车辆碰撞、失速等。| - |
| 医学图像分析 | [AGEchallenge](https://github.com/PaddlePaddle/Research/CV/AGEchallenge) | 任务：在AS-OCT图像的公共数据集上进行闭角型分类和巩膜突点定位；基线模型：对应以上各任务的基线模型。 | - |
| 光流估计 | [PWCNet](https://github.com/PaddlePaddle/Research/CV/PWCNet) | 基于金字塔式处理，逐层学习细部光流，设计代价容量函数三原则的CNN模型，用于光流估计。 | https://arxiv.org/abs/1709.02371 |
| 语义分割 | [SemSegPaddle](https://github.com/PaddlePaddle/Research/CV/SemSegPaddle) | 针对多个数据集的图像语义分割模型的实现，包括Cityscapes、Pascal Context和ADE20K。 | - |
| 轻量化检测 | [astar2019](https://github.com/PaddlePaddle/Research/CV/astar2019) | 百度之星轻量化检测比赛评测工具。 | - |
| 地标检索与识别 | [landmark](https://github.com/PaddlePaddle/Research/CV/landmark) | 基于检索的地标检索与识别系统，支持地标型与非地标型识别、识别与检索结果相结合的多重识别结果投票和重新排序。 | https://arxiv.org/abs/1906.03990 |
| 图像分类 | [webvision2018](https://github.com/PaddlePaddle/Research/CV/webvision2018) | 模型利用重加权网络(URNet)缓解web数据中偏倚和噪声的影响，进行web图像分类。 | https://arxiv.org/abs/1811.00700 |
| 图像分类 | [CLPI](https://github.com/PaddlePaddle/Research/CV/CLPI-Collaborative-Learning-for-Diabetic-Retinopathy-Grading) | 模型利用一个Lesion Generator改善了糖尿病视网膜病变图像分级的模型性能，理论上可用于所有希望实现局部+整体模型分析的场景 | - |
| 小样本学习 | [PaddleFSL](https://github.com/PaddlePaddle/Research/CV/PaddleFSL) | 小样本学习工具包，可复现多个常用基线方法在多个图片分类数据集上的汇报效果 | - |

## 自然语言处理
| 任务类型     | 目录                                                         | 简介                                                         | 论文链接 |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| 中文词法分析 | [LAC(Lexical Analysis of Chinese)](https://github.com/baidu/lac) | 百度自主研发中文特色模型词法分析任务，集成了中文分词、词性标注和命名实体识别任务。输入是一个字符串，而输出是句子中的词边界和词性、实体类别。 | - |
| 主动对话 | [DuConv](https://github.com/PaddlePaddle/Research/NLP/ACL2019-DuConv) | 机器根据给定知识信息主动引领对话进程完成设定的对话目标。 |https://www.aclweb.org/anthology/P19-1369/|
| 语义解析 | [Text2SQL-BASELINE](https://github.com/PaddlePaddle/Research/NLP/Text2SQL-BASELINE) | 输入自然语言问题和相应的数据库，生成与问题对应的 SQL 查询语句，通过执行该 SQL 可得到问题的答案。 | - |
| 多轮对话 | [DAM](https://github.com/PaddlePaddle/Research/NLP/ACL2018-DAM) | 开放领域多轮对话匹配的深度注意力机制模型，根据多轮对话历史和候选回复内容，排序出最合适的回复。 | http://aclweb.org/anthology/P18-1103 |
| 阅读理解 | [DuReader](https://github.com/PaddlePaddle/Research/NLP/ACL2018-DuReader) | 数据集：大规模、面向真实应用、由人类生成的中文阅读理解数据集，聚焦于真实世界中的不限定领域的问答任务；基线系统：针对DuReader数据集实现的经典BiDAF模型。 | https://www.aclweb.org/anthology/W18-2605/ |
| 关系抽取 | [ARNOR](https://github.com/PaddlePaddle/Research/NLP/ACL2019-ARNOR) | 数据集：用于对远程监督关系提取模型进行句子级别的评价；模型：基于注意力正则化识别噪声数据，通过bootstrap方法逐步选择出高质量的标注数据。| https://www.aclweb.org/anthology/P19-1135/ |
| 机器翻译 | [JEMT](https://github.com/PaddlePaddle/Research/NLP/ACL2019-JEMT) | 模型的输入端包括文字信息及发音信息，嵌入层融合文字信息和发音信息进行翻译。 | https://arxiv.org/abs/1810.06729 |
| 阅读理解 | [KTNET](https://github.com/PaddlePaddle/Research/NLP/ACL2019-KTNET) | 模型将知识库中的知识整合到预先训练好的上下文表示中，利用丰富的知识增强机器阅读理解的预训练语言表示。 | https://www.aclweb.org/anthology/P19-1226 |
| 对话生成 | [PLATO](https://github.com/PaddlePaddle/Research/NLP/Dialogue-PLATO) | 基于隐空间的端到端的预训练对话生成模型，可以灵活支持多种对话，包括闲聊、知识聊天、对话问答等。 | http://arxiv.org/abs/1910.07931 |
| 阅读理解 | [DuReader-Robust-BASELINE](https://github.com/PaddlePaddle/Research/NLP/DuReader-Robust-BASELINE) | 数据集：DuReader-robust，中文数据集，用于全面评价机器阅读理解模型的鲁棒性；基线系统：针对该数据集，基于[ERNIE](https://arxiv.org/abs/1904.09223)实现的阅读理解基线系统。 | https://arxiv.org/abs/2004.11142 |
| 对话生成 | [AKGCM](https://github.com/PaddlePaddle/Research/NLP/EMNLP2019-AKGCM) | 包含知识增强图、知识选择和知识感知响应生成器的聊天机器人。 | https://www.aclweb.org/anthology/D19-1187/ |
| 机器翻译 | [MAL](https://github.com/PaddlePaddle/Research/NLP/EMNLP2019-MAL) | 多智能体端到端联合学习框架，通过多个智能体的互相学习提升翻译质量。 | https://arxiv.org/abs/1909.01101 |
| 对话生成 | [MMPMS](https://github.com/PaddlePaddle/Research/NLP/IJCAI2019-MMPMS) | 针对开放域对话中一对多问题，利用多映射机制和后验映射选择模块进行多样性、丰富化的对话生成。 | https://arxiv.org/abs/1906.01781 |
| 阅读理解 | [MRQA2019-BASELINE](https://github.com/PaddlePaddle/Research/NLP/MRQA2019-BASELINE) | 机器阅读理解任务的基线模型，基于[ERNIE](https://arxiv.org/abs/1904.09223)预训练模型，支持多GPU微调预测。 | - |
| 阅读理解 | [D-NET](https://github.com/PaddlePaddle/Research/NLP/MRQA2019-D-NET) | 预训练及微调框架，包含多任务学习及多预训练模型的融合，用于阅读理解模型的生成。 | https://www.aclweb.org/anthology/D19-5828/ |
| 建议挖掘 | [MPM](https://github.com/PaddlePaddle/Research/NLP/NAACL2019-MPM) | 利用多视角架构来学习表示和双向transformer编码器进行论坛评论建议挖掘。 | https://www.aclweb.org/anthology/S19-2216/ |
| 多文档摘要 | [ACL2020-GraphSum](https://github.com/PaddlePaddle/Research/NLP/ACL2020-GraphSum) | 基于图表示的生成式多文档摘要模型，将显式图结构信息引入到端到端摘要生成过程中。 | https://www.aclweb.org/anthology/2020.acl-main.555.pdf |
| 融合多种对话类型的对话式推荐 | [ACL2020-DuRecDial](https://github.com/PaddlePaddle/Research/NLP/ACL2020-DuRecDial) | 提出新任务：融合闲聊、任务型对话、问答和推荐等多种对话类型的对话式推荐，构建DuRecDial数据集，提出具有多对话目标驱动策略机制的对话生成框架。 | https://www.aclweb.org/anthology/2020.acl-main.98/ |
| 面向推荐的对话 | [Conversational-Recommendation-BASELINE](https://github.com/PaddlePaddle/Research/NLP/Conversational-Recommendation-BASELINE) | 融合人机对话系统和个性化推荐系统，定义新一代智能推荐技术，该系统先通过问答或闲聊收集用户兴趣和偏好，然后主动给用户推荐其感兴趣的内容，比如餐厅、美食、电影、新闻等。 | - |
| 稠密段落检索 | [ACL2021-PAIR](https://github.com/PaddlePaddle/Research/NLP/ACL2021-PAIR) | 基于以段落相似度为中心的相似度关系提升稠密段落检索，基于知识蒸馏进行采样，采用两阶段训练方式。 | https://aclanthology.org/2021.findings-acl.191/ |

## 知识图谱
| 任务类型     | 目录                                                         | 简介                                                         | 论文链接 |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| 知识图谱表示学习 | [CoKE](https://github.com/PaddlePaddle/Research/KG/CoKE) | 百度自主研发语境化知识图谱表示学习框架CoKE，在知识图谱链接预测和多步查询任务上取得学界领先效果。| [https://arxiv.org/abs/1911.02168](https://arxiv.org/abs/1911.02168) |
| 关系抽取 | [DuIE\_Baseline](https://github.com/PaddlePaddle/Research/KG/DuIE_Baseline) | 语言与智能技术竞赛关系抽取任务DuIE 2.0基线系统，通过设计结构化标注体系，实现基于[ERNIE](https://arxiv.org/abs/1904.09223)的端到端SPO抽取模型。| - |
| 事件抽取 |[DuEE\_baseline](https://github.com/PaddlePaddle/Research/KG/DuEE_baseline)| 语言与智能技术竞赛事件抽取任务DuEE 1.0基线系统，实现基于[ERNIE](https://arxiv.org/abs/1904.09223)+CRF的Pipeline事件抽取模型。| - |
| 实体链指 |[DuEL\_Baseline](https://github.com/PaddlePaddle/Research/KG/DuEL_Baseline)| 面向中文短文本的实体链指任务(CCKS 2020)的基线系统，实现基于[ERNIE](https://arxiv.org/abs/1904.09223)和多任务机制的实体链指模型。| - |
| 辅助诊断 |[SignOrSymptom\_Relationship](https://github.com/PaddlePaddle/Research/KG/ACL2020_SignOrSymptom_Relationship)| 针对EMR具有无结构化文本和结构化信息并存的特点，结合医疗NLU，以深度学习模型实现EMR的向量化表示、诊断预分类和概率计算。| - |
| 文档级关系抽取 | [SSAN](https://github.com/PaddlePaddle/Research/KG/AAAI2021_SSAN) | 引入并建模实体间的依赖结构，在文档级关系抽取任务上取得学界领先效果。| [https://arxiv.org/abs/2102.10249](https://arxiv.org/abs/2102.10249) |

## 时空数据挖掘
| 任务类型     | 目录                                                         | 简介                                                         | 论文链接 |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| 固定资产价值估计 |[MONOPOLY](https://github.com/PaddlePaddle/Research/ST_DM/CIKM2019-MONOPOLY)| 实用的POI商业智能算法，对大量其他的固定资产进行价值估计，包括城市居民对不同公共资产价格评估、私有房价评估偏好的发现与量化分析，以及对评估固定资产价格需考虑的空间范围的确定。 | https://dl.acm.org/doi/10.1145/3357384.3357810 |
| 兴趣点生成 |[P3AC](https://github.com/PaddlePaddle/Research/ST_DM/KDD2020-P3AC)| 具备个性化的前缀嵌入的POI自动生成。 | - |
| 区域生成 |[P3AC](https://github.com/PaddlePaddle/Research/ST_DM/GenRegion)| 基于路网进行区域划分的方法, 实现对特定区域基于路网的全划分，区域之间无交叠，无空隙，算法支持对全球的区域划分。| - |


## 许可证书
此向导由[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)贡献，受[Apache-2.0 license](LICENSE)许可认证。