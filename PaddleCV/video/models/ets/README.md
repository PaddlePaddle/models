# ETS 视频描述模型

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [模型推断](#模型推断)
- [参考论文](#参考论文)


## 模型简介

Describing Videos by Exploiting Temporal Structure是由蒙特利尔大学Li Yao等人提出的用于对视频片段生成文字描述的经典模型，这里简称为ETS。此模型基于编码器-解码器的思想，对输入的视频，先使用3D卷积提取视频的局部时空特征，然后在时序维度上引入注意力机制，利用LSTM在全局尺度上对局部特征进行融合，最后输出文字描述。

详细内容请参考[Describing Videos by Exploiting Temporal Structure](https://arxiv.org/abs/1502.08029)。


## 数据准备

ETS的训练数据采用ActivityNet Captions提供的数据集，数据下载及准备请参考[数据说明](../../data/dataset/ets/README.md)

## 模型训练

数据准备完毕后，可以通过如下两种方式启动训练：

    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export FLAGS_fast_eager_deletion_mode=1
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=0.98
    python train.py --model_name=ETS \
                    --config=./configs/ets.yaml \
                    --log_interval=10 \
                    --valid_interval=1 \
                    --use_gpu=True \
                    --save_dir=./data/checkpoints \
                    --fix_random_seed=False

    bash run.sh train ETS ./configs/ets.yaml

- 从头开始训练，使用上述启动命令行或者脚本程序即可启动训练，不需要用到预训练模型

- 可下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_caption/ETS.pdparams)通过`--resume`指定权重存放路径进行finetune等开发


**训练策略：**

*  采用Adam优化算法训练
*  权重衰减系数为1e-4
*  学习率调整使用Noam衰减方法

## 模型评估

可通过如下两种方式进行模型评估:

    python eval.py --model_name=ETS \
                   --config=./configs/ets.yaml \
                   --log_interval=1 \
                   --weights=$PATH_TO_WEIGHTS \
                   --use_gpu=True

    bash run.sh eval ETS ./configs/ets.yaml

- 使用`run.sh`进行评估时，需要修改脚本中的`weights`参数指定需要评估的权重。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_caption/ETS.pdparams)进行评估

- 运行上述程序会将测试结果保存在json文件中，默认存储在data/evaluate\_results目录下。使用ActivityNet Captions官方提供的测试脚本，即可计算METEOR。具体计算过程请参考[指标计算](../../metrics/ets_metrics/README.md)

- 使用CPU进行评估时，请将上面的命令行或者run.sh脚本中的`use_gpu`设置为False


在ActivityNet Captions数据集下评估精度如下:

| METEOR |
| :----: |
|  9.8  |


## 模型推断

可通过如下两种方式启动模型推断：

    python predict.py --model_name=ETS \
                      --config=./configs/ets.yaml \
                      --log_interval=1 \
                      --weights=$PATH_TO_WEIGHTS \
                      --filelist=$FILELIST \
                      --use_gpu=True

    bash run.sh predict ETS ./configs/ets.yaml

- 使用python命令行启动程序时，`--filelist`参数指定待推断的文件列表。用户也可参考[数据说明](../../data/dataset/ets/README.md)步骤三生成默认的推断文件列表。`--weights`参数为训练好的权重参数，如果不设置，程序会自动下载已训练好的权重。

- 使用`run.sh`进行评估时，需要修改脚本中的`weights`参数指定需要用到的权重。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_caption/ETS.pdparams)进行推断

- 模型推断结果存储于json文件中，默认存储在`data/dataset/predict_results`目录下

- 使用CPU进行推断时，请将命令行或者run.sh脚本中的`use_gpu`设置为False

## 参考论文

- [Describing Videos by Exploiting Temporal Structure](https://arxiv.org/abs/1502.08029), Li Yao, Atousa Torabi, Kyunghyun Cho, Nicolas Ballas, Christopher Pal, Hugo Larochelle, Aaron Courville.
