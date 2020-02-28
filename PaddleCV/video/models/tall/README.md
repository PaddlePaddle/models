# TALL 视频查找模型

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [模型推断](#模型推断)
- [参考论文](#参考论文)


## 模型简介

TALL是由南加州大学的Jiyang Gao等人提出的视频查找方向的经典模型。对输入的文本序列和视频片段，TALL模型利用多模态时序回归定位器(Cross-modal Temporal Regression Localizer, CTRL)联合视频信息和文本描述信息，输出位置偏置和置信度。CTRL包含四个模块：视觉编码器从视频片段中提取特征，文本编码器从语句中提取特征向量，多模态处理网络结合文本和视觉特征生成联合特征，最后时序回归网络生成置信度和偏置。

详细内容请参考[TALL: Temporal Activity Localization via Language Query](https://arxiv.org/abs/1705.02101)。


## 数据准备

TALL的训练数据采用TACoS数据集，数据下载及准备请参考[数据说明](../../data/dataset/tall/README.md)

## 模型训练

数据准备完毕后，可以通过如下两种方式启动训练：

    export CUDA_VISIBLE_DEVICES=0
    export FLAGS_fast_eager_deletion_mode=1
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=0.98
    python train.py --model_name=TALL \
                    --config=./configs/tall.yaml \
                    --log_interval=10 \
                    --valid_interval=10000 \
                    --use_gpu=True \
                    --save_dir=./data/checkpoints \
                    --fix_random_seed=False

    bash run.sh train TALL ./configs/tall.yaml

- 从头开始训练，使用上述启动命令行或者脚本程序即可启动训练，不需要用到预训练模型

- 可下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_grounding/TALL.pdparams)通过`--resume`指定权重存放路径进行finetune等开发

- 模型未设置验证集，故将valid\_interval设为10000，在训练过程中不进行验证。


**训练策略：**

*  采用Adam优化算法训练
*  学习率为1e-3

## 模型评估

可通过如下两种方式进行模型评估:

    python eval.py --model_name=TALL \
                   --config=./configs/tall.yaml \
                   --log_interval=1 \
                   --weights=$PATH_TO_WEIGHTS \
                   --use_gpu=True

    bash run.sh eval TALL ./configs/tall.yaml

- 使用`run.sh`进行评估时，需要修改脚本中的`weights`参数指定需要评估的权重。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_grounding/TALL.pdparams)进行评估

- 运行上述程序会将测试结果打印出来，同时保存在json文件中，默认存储在data/evaluate\_results目录下。

- 使用CPU进行评估时，请将上面的命令行或者run.sh脚本中的`use_gpu`设置为False


在TACoS数据集下评估精度如下:

| R1@IOU5 | R5@IOU5 |
| :----: | :----: |
|  0.13  |  0.24  |


## 模型推断

可通过如下两种方式启动模型推断：

    python predict.py --model_name=TALL \
                      --config=./configs/tall.yaml \
                      --log_interval=1 \
                      --weights=$PATH_TO_WEIGHTS \
                      --filelist=$FILELIST \
                      --use_gpu=True

    bash run.sh predict TALL ./configs/tall.yaml

- 使用python命令行启动程序时，`--filelist`参数指定待推断的文件列表。用户也可参考[数据说明](../../data/dataset/tall/README.md)步骤二生成默认的推断文件。`--weights`参数为训练好的权重参数，如果不设置，程序会自动下载已训练好的权重。

- 使用`run.sh`进行评估时，需要修改脚本中的`weights`参数指定需要用到的权重。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_grounding/TALL.pdparams)进行推断

- 模型推断结果存储于json文件中，默认存储在`data/dataset/predict_results`目录下。

- 使用CPU进行推断时，请将命令行或者run.sh脚本中的`use_gpu`设置为False

## 参考论文

- [TALL: Temporal Activity Localization via Language Query](https://arxiv.org/abs/1705.02101), Jiyang Gao, Chen Sun, Zhenheng Yang, Ram Nevatia.
