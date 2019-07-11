# C-TCN 视频动作定位模型

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [模型推断](#模型推断)
- [参考论文](#参考论文)


## 模型简介

C-TCN动作定位模型是百度自研，2018年ActivityNet夺冠方案，在Paddle上首次开源，为开发者提供了处理视频动作定位问题的解决方案。此模型引入了concept-wise时间卷积网络，对每个concept先用卷积神经网络分别提取时间维度的信息，然后再将每个concept的信息进行组合。主体结构是残差网络+FPN，采用类似SSD的单阶段目标检测算法对时间维度的anchor box进行预测和分类。


## 数据准备

C-TCN的训练数据采用ActivityNet1.3提供的数据集，数据下载及准备请参考[数据说明](../../dataset/ctcn/README.md)

## 模型训练

数据准备完毕后，可以通过如下两种方式启动训练：

    python train.py --model_name=CTCN
            --config=./configs/ctcn.txt
            --save_dir=checkpoints
            --log_interval=10
            --valid_interval=1
            --pretrain=${path_to_pretrain_model}

    bash scripts/train/train_ctcn.sh

- 从头开始训练，使用上述启动脚本程序即可启动训练，不需要用到预训练模型

- 可下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_detection/ctcn.tar.gz)通过`--resume`指定权重存放路径进行finetune等开发


**训练策略：**

*  采用Momentum优化算法训练，momentum=0.9
*  权重衰减系数为1e-4
*  学习率在迭代次数达到9000的时候做一次衰减

## 模型评估

可通过如下两种方式进行模型评估:

    python test.py --model_name=CTCN
            --config=configs/ctcn.txt
            --log_interval=1
            --weights=$PATH_TO_WEIGHTS

    bash scripts/test/test_ctcn.sh

- 使用`scripts/test/test_ctcn.sh`进行评估时，需要修改脚本中的`--weights`参数指定需要评估的权重。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_detection/ctcn.tar.gz)进行评估

- 运行上述程序会将测试结果保存在json文件中，使用ActivityNet官方提供的测试脚本，即可计算MAP。具体计算过程请参考[指标计算](../../metrics/detections/README.md)

当取如下参数时，在ActivityNet1.3数据集下评估精度如下:

| score\_thresh | nms\_thresh | soft\_sigma | soft\_thresh | MAP |
| :-----------: | :---------: | :---------: | :----------: | :---: |
| 0.001 | 0.8 | 0.9 | 0.004 | 31% |


## 模型推断

可通过如下命令进行模型推断：

    python infer.py --model_name=CTCN
            --config=configs/ctcn.txt
            --log_interval=1
            --weights=$PATH_TO_WEIGHTS
            --filelist=$FILELIST

- 模型推断结果存储于`CTCN_infer_result.pkl`中，通过`pickle`格式存储。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_detection/ctcn.tar.gz)进行推断

## 参考论文

- 待发表
