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

C-TCN动作定位模型是百度自研，2018年ActivityNet夺冠方案，在PaddlePaddle上首次开源，为开发者提供了处理视频动作定位问题的解决方案。此模型引入了concept-wise时间卷积网络，对每个concept先用卷积神经网络分别提取时间维度的信息，然后再将每个concept的信息进行组合。主体结构是残差网络+FPN，采用类似SSD的单阶段目标检测算法对时间维度的anchor box进行预测和分类。


## 数据准备

C-TCN的训练数据采用ActivityNet1.3提供的数据集，数据下载及准备请参考[数据说明](../../data/dataset/ctcn/README.md)

## 模型训练

数据准备完毕后，可以通过如下两种方式启动训练：

    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export FLAGS_fast_eager_deletion_mode=1
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=0.98
    python train.py --model_name=CTCN \
                    --config=./configs/ctcn.yaml \
                    --log_interval=10 \
                    --valid_interval=1 \
                    --use_gpu=True \
                    --save_dir=./data/checkpoints \
                    --fix_random_seed=False \
                    --pretrain=$PATH_TO_PRETRAIN_MODEL

    bash run.sh train CTCN ./configs/ctcn.yaml

- 从头开始训练，使用上述启动命令行或者脚本程序即可启动训练，不需要用到预训练模型

- 可下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_detection/CTCN.pdparams)通过`--resume`指定权重存放路径进行finetune等开发


**训练策略：**

*  采用Momentum优化算法训练，momentum=0.9
*  权重衰减系数为1e-4
*  学习率在迭代次数达到9000的时候做一次衰减

## 模型评估

可通过如下两种方式进行模型评估:

    python eval.py --model_name=CTCN \
                   --config=./configs/ctcn.yaml \
                   --log_interval=1 \
                   --weights=$PATH_TO_WEIGHTS \
                   --use_gpu=True

    bash run.sh eval CTCN ./configs/ctcn.yaml

- 使用`run.sh`进行评估时，需要修改脚本中的`weights`参数指定需要评估的权重。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_detection/CTCN.pdparams)进行评估

- 运行上述程序会将测试结果保存在json文件中，默认存储在data/evaluate\_results目录下，程序根据所使用的超参数自动生成文件名，例如：CTCN\_test\_res\_decode\_0.001\_0.8\_0.9\_0.004.json。使用ActivityNet官方提供的测试脚本，即可计算MAP。具体计算过程请参考[指标计算](../../metrics/detections/README.md)

- 使用CPU进行评估时，请将上面的命令行或者run.sh脚本中的`use_gpu`设置为False


当取如下参数时，在ActivityNet1.3数据集下评估精度如下:

| score\_thresh | nms\_thresh | soft\_sigma | soft\_thresh | MAP |
| :-----------: | :---------: | :---------: | :----------: | :---: |
| 0.001 | 0.8 | 0.9 | 0.004 | 31% |


## 模型推断

可通过如下两种方式启动模型推断：

    python predict.py --model_name=CTCN \
                      --config=./configs/ctcn.yaml \
                      --log_interval=1 \
                      --weights=$PATH_TO_WEIGHTS \
                      --filelist=$FILELIST \
                      --use_gpu=True

    bash run.sh predict CTCN ./configs/ctcn.yaml

- 使用python命令行启动程序时，`--filelist`参数指定待推断的文件列表，如果不设置，默认为data/dataset/youtube8m/infer.list。`--weights`参数为训练好的权重参数，如果不设置，程序会自动下载已训练好的权重。这两个参数如果不设置，请不要写在命令行，将会自动使用默
认值。

- 使用`run.sh`进行评估时，需要修改脚本中的`weights`参数指定需要用到的权重。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_detection/CTCN.pdparams)进行推断


- 模型推断结果存储于json文件中，默认存储在`data/dataset/inference_results`目录下，程序根据所使用的超参数自动生成文件名，例如：CTCN\_infer\_res\_decode\_0.001\_0.8\_0.9\_0.004.json。同时也会以log的形式打印输出，显示每个视频的预测片段起止时间和类别

- 使用CPU进行推断时，请将命令行或者run.sh脚本中的`use_gpu`设置为False

## 参考论文

- 待发表
