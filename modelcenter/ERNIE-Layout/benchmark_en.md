# Benchmark

## 1.Software And Hardware Environment
The training and reasoning of ERNIE-Layout model adopt Tesla V100-SXM2-16GB, CUDA 10.2, CUDNN 7.5.1, and paddlepaddle-gpu 2.3.2.

## 2.Introduction To Open Source Datasets
| Dataset | Task Type | Language | Explain |
| ---- | ---- | ---- | ----|
| FUNSD | Document information extraction | English | - |
| XFUND-ZH | Document information extraction | Chinese | - |
| DocVQA-ZH | Document Visual Q&A | Chinese | [DocVQA-ZH](http://ailab.aiwin.org.cn/competitions/49) has stopped submitting the list, so we will re divide the original training set to evaluate the model effect. After division, the training set contains 4187 images, the verification set contains 500 images, and the test set contains 500 images. |
| RVL-CDIP (sampled) | Document Image Classification | English | The RVL-CDIP original data set contains 400000 pictures in total. Because the data set is large and the training is slow, the sampling is reduced to verify the model effect of document image classification. The sampled training set contains 6400 pictures, the verification set contains 800 pictures, and the test set contains 800 pictures. |

## 3.Evaluation Results
The evaluation indicators on the **validation set** of mainstream open source datasets in the field of document intelligence are shown in the following table:
| Model | FUNSD | RVL-CDIP (sampled) | XFUND-ZH | DocVQA-ZH |
| ---- | ---- | ---- | ---- | ---- |
| LayoutXLM-Base | 86.72 | 90.88 | 86.24 | 66.01 |
| ERNIE-LayoutX-Base | 89.31 | 90.29 | 88.58 | 69.57 |

## 4.Specific Evaluation Method
* All the above tasks are based on the Grid Search method for super parameter optimization. FUNSD and XFUND-ZH evaluate the effect of validation set every 100 steps, and the evaluation index is F1-Score. RVL-CDIP evaluates the effect of validation set every 2000 steps, and the evaluation index is Accuracy. DocVQA-ZH evaluates the effect of the validation set every 10000 steps, and takes the best effect of the validation set as the reporting indicator in the table. The evaluation indicator is ANLS (refer to the calculation method [ICDAR 2019 Competition on Scene Text Visual Question Answering](https://arxiv.org/pdf/1907.00490.pdf))
* The super parameter range of each downstream task above is shown in the following table:

| Hyper Parameters | FUNSD | RVL-CDIP (sampled) | XFUND-ZH | DocVQA-ZH |
| ---- | ---- | ---- | ---- | ---- |
| learning_rate | 5e-6, 1e-5, 2e-5, 5e-5 | 5e-6, 1e-5, 2e-5, 5e-5 | 5e-6, 1e-5, 2e-5, 5e-5 | 5e-6, 1e-5, 2e-5, 5e-5 |
| batch_size | 1, 2, 4 | 8, 16, 24 | 1, 2, 4 | 8, 16, 24 |
| warmup_ratio | - | 0, 0.05, 0.1 | - | 0, 0.05, 0.1 |

<figure>
The lr_scheduler_type policy used by FUNSD and XFUND-ZH is constant, so it is not warmup_ratio to search.
</figure>

* The document information extraction tasks FUNSD and XFUND-ZH adopt the maximum steps (max_steps) fine-tuning method, which are 10000 steps and 20000 steps respectively; Document Visual Q&A DocVQA-ZH num_train_epochs is 6; Document image classification RVL-CDIP num_train_epochs is 20.


* Optimal hyperparameter 
Optimal hyperparameters of different pre training models after Grid Search on downstream tasks （learning_rate、batch_size、warmup_ratio）are as follows： 


| Model | FUNSD | RVL-CDIP (sampled) | XFUND-ZH | DocVQA-ZH |
| ---- | ---- | ---- | ---- | ---- |
| LayoutXLM-Base | 1e-5, 2, _ | 1e-5, 8, 0.1 | 1e-5, 2, _ | 2e-5. 8, 0.1 |
| ERNIE-LayoutX-Base | 2e-5, 4, _ | 1e-5, 8, 0. | 1e-5, 4, _ | 2e-5. 8, 0.05 |

# 5.Relevant Instructions
Please refer to：[ERNIE-Layout](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/ernie-layout/README_ch.md)