# Benchmark

## 1.软硬件环境
ERNIE-Layout模型的训练和推理均采用Tesla V100-SXM2-16GB、CUDA 10.2、 CUDNN 7.5.1、paddlepaddle-gpu 2.3.2.

## 2.开源数据集介绍
| 数据集 | 任务类型 | 语言 | 说明 |
| ---- | ---- | ---- | ----|
| FUNSD | 文档信息抽取 | 英文 | - |
| XFUND-ZH | 文档信息抽取 | 中文 | - |
| DocVQA-ZH | 文档视觉问答 | 中文 | [DocVQA-ZH](http://ailab.aiwin.org.cn/competitions/49)已停止榜单提交，因此我们将原始训练集进行重新划分以评估模型效果，划分后训练集包含4,187张图片，验证集包含500张图片，测试集包含500张图片。 |
| RVL-CDIP (sampled) | 文档图像分类 | 英文 | RVL-CDIP原始数据集共包含400,000张图片，由于数据集较大训练较慢，为验证文档图像分类的模型效果故进行降采样，采样后的训练集包含6,400张图片，验证集包含800张图片，测试集包含800张图片。 |

## 3.评测结果
在文档智能领域主流开源数据集的**验证集**上评测指标如下表所示：
| Model | FUNSD | RVL-CDIP (sampled) | XFUND-ZH | DocVQA-ZH |
| ---- | ---- | ---- | ---- | ---- |
| LayoutXLM-Base | 86.72 | 90.88 | 86.24 | 66.01 |
| ERNIE-LayoutX-Base | 89.31 | 90.29 | 88.58 | 69.57 |

## 4.具体评测方式
* 以上所有任务均基于Grid Search方式进行超参寻优。FUNSD和XFUND-ZH每间隔 100 steps 评估验证集效果，评价指标为F1-Score。 RVL-CDIP每间隔2000 steps评估验证集效果，评价指标为Accuracy。DocVQA-ZH每间隔10000 steps评估验证集效果，取验证集最优效果作为表格中的汇报指标，评价指标为ANLS(计算方法参考[ICDAR 2019 Competition on Scene Text Visual Question Answering)
  ](https://arxiv.org/pdf/1907.00490.pdf)
* 以上每个下游任务的超参范围如下表所示：

| Hyper Parameters | FUNSD | RVL-CDIP (sampled) | XFUND-ZH | DocVQA-ZH |
| ---- | ---- | ---- | ---- | ---- |
| learning_rate | 5e-6, 1e-5, 2e-5, 5e-5 | 5e-6, 1e-5, 2e-5, 5e-5 | 5e-6, 1e-5, 2e-5, 5e-5 | 5e-6, 1e-5, 2e-5, 5e-5 |
| batch_size | 1, 2, 4 | 8, 16, 24 | 1, 2, 4 | 8, 16, 24 |
| warmup_ratio | - | 0, 0.05, 0.1 | - | 0, 0.05, 0.1 |

<figure>
FUNSD和XFUND-ZH使用的lr_scheduler_type策略是constant，因此不对warmup_ratio进行搜索。
</figure>

* 文档信息抽取任务FUNSD和XFUND-ZH采用最大步数（max_steps）的微调方式，分别为10000 steps和20000 steps；文档视觉问答DocVQA-ZH的num_train_epochs为6；文档图像分类RVL-CDIP的num_train_epochs为20。


* 最优超参  
不同预训练模型在下游任务上做Grid Search之后的最优超参（learning_rate、batch_size、warmup_ratio）如下： 


| Model | FUNSD | RVL-CDIP (sampled) | XFUND-ZH | DocVQA-ZH |
| ---- | ---- | ---- | ---- | ---- |
| LayoutXLM-Base | 1e-5, 2, _ | 1e-5, 8, 0.1 | 1e-5, 2, _ | 2e-5. 8, 0.1 |
| ERNIE-LayoutX-Base | 2e-5, 4, _ | 1e-5, 8, 0. | 1e-5, 4, _ | 2e-5. 8, 0.05 |

# 5.相关使用说明
请参考：[ERNIE-Layout](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/ernie-layout/README_ch.md)