**模型简介**

图像分类是计算机视觉的重要领域，它的目标是将图像分类到预定义的标签。CNN模型在图像分类领域取得了突破的成果，同时模型复杂度也在不断增加。MobileNet是一种小巧而高效CNN模型，本文介绍如何使PaddlePaddle的动态图MobileNet进行图像分类。

**代码结构**

    ├── run_mul_v1.sh                 # 多卡训练启动脚本_v1
    ├── run_mul_v1_checkpoint.sh      # 加载checkpoint多卡训练启动脚本_v1
    ├── run_mul_v2.sh                 # 多卡训练启动脚本_v2
    ├── run_mul_v2_checkpoint.sh      # 加载checkpoint多卡训练启动脚本_v2
    ├── run_sing_v1.sh                # 单卡训练启动脚本_v1
    ├── run_sing_v1_checkpoint.sh     # 加载checkpoint单卡训练启动脚本_v1
    ├── run_sing_v2.sh                # 单卡训练启动脚本_v2
    ├── run_sing_v2_checkpoint.sh     # 加载checkpoint单卡训练启动脚本_v2
    ├── run_cpu_v1.sh                 # CPU训练启动脚本_v1
    ├── run_cpu_v2.sh                 # CPU训练启动脚本_v2
    ├── train.py                      # 训练入口
    ├── mobilenet_v1.py               # 网络结构v1
    ├── mobilenet_v2.py               # 网络结构v2
    ├── reader.py                     # 数据reader
    ├── utils                         # 基础工具目录

**数据准备**

请参考：https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification

**模型训练**

若使用4卡训练，启动方式如下:

    bash run_mul_v1.sh
    bash run_mul_v2.sh

若使用单卡训练，启动方式如下:

    bash run_sing_v1.sh
    bash run_sing_v2.sh

若使用CPU训练，启动方式如下:

    bash run_cpu_v1.sh
    bash run_cpu_v2.sh

训练过程中,checkpoint会保存在参数model_save_dir指定的文件夹中,我们支持加载checkpoint继续训练.
加载checkpoint使用4卡训练，启动方式如下:

    bash run_mul_v1_checkpoint.sh
    bash run_mul_v2_checkpoint.sh

加载checkpoint使用单卡训练，启动方式如下:

    bash run_sing_v1_checkpoint.sh
    bash run_sing_v2_checkpoint.sh

**模型性能**

    Model          Top-1(单卡/4卡)    Top-5(单卡/4卡)    收敛时间(单卡/4卡)
    
    MobileNetV1    0.707/0.711        0.897/0.899        116小时/30.9小时
    
    MobileNetV2    0.708/0.724        0.899/0.906        227.8小时/60.8小时

**参考论文**

MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam

MobileNetV2: Inverted Residuals and Linear Bottlenecks, Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
