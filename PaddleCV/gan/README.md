# 图像生成模型库

生成对抗网络(Generative Adversarial Network\[[1](#参考文献)\], 简称GAN) 是一种非监督学习的方式，通过让两个神经网络相互博弈的方法进行学习，该方法由lan Goodfellow等人在2014年提出。生成对抗网络由一个生成网络和一个判别网络组成，生成网络从潜在的空间(latent space)中随机采样作为输入，其输出结果需要尽量模仿训练集中的真实样本。判别网络的输入为真实样本或生成网络的输出，其目的是将生成网络的输出从真实样本中尽可能的分辨出来。而生成网络则尽可能的欺骗判别网络，两个网络相互对抗，不断调整参数。
生成对抗网络常用于生成以假乱真的图片。此外，该方法还被用于生成影片，三维物体模型等。\[[2](#参考文献)\]

---
## 内容

-[简介](#简介)

-[快速开始](#快速开始)

-[参考文献](#参考文献)

## 简介

本图像生成模型库包含CGAN\[[3](#参考文献)\], DCGAN\[[4](#参考文献)\], Pix2Pix\[[5](#参考文献)\], CycleGAN\[[6](#参考文献)\], StarGAN\[[7](#参考文献)\], AttGAN\[[8](#参考文献)\], STGAN\[[9](#参考文献)\]。

图像生成模型库库的目录结构如下：
```
├── download.py 下载数据
│  
├── data_reader.py 数据预处理
│  
├── train.py 模型的训练入口
│  
├── infer.py 模型的预测入口
│  
├── trainer 不同模型的训练脚本
│   ├── CGAN.py Conditional GAN的训练脚本
│   ├── DCGAN.py Deep Convolutional GAN的训练脚本
│   ├── Pix2pix.py Pix2Pix GAN的训练脚本
│   ├── CycleGAN.py CycleGAN的训练脚本
│   ├── StarGAN.py StarGAN的训练脚本
│   ├── AttGAN.py AttGAN的训练脚本
│   ├── STGAN.py STGAN的训练脚本
│  
├── network 不同模型的网络结构
│   ├── base_network.py GAN模型需要的公共基础网络结构
│   ├── CGAN_network.py Conditional GAN的网络结构
│   ├── DCGAN_network.py Deep Convolutional GAN的网络结构
│   ├── Pix2pix_network.py Pix2Pix GAN的网络结构
│   ├── CycleGAN_network.py CycleGAN的网络结构
│   ├── StarGAN_network.py StarGAN的网络结构
│   ├── AttGAN_network.py AttGAN的网络结构
│   ├── STGAN_network.py STGAN的网络结构
│  
├── util 网络的基础配置和公共模块
│   ├── config.py 网络公用的基础配置
│   ├── utility.py 保存模型等网络公用的模块
│  
├── scripts 多个模型的训练启动和测试启动示例
│   ├── run_....py 训练启动示例
│   ├── infer_....py 测试启动示例
│   ├── make_pair_data.py pix2pix GAN的数据list的生成脚本

```

## 快速开始
**安装[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)：**

在当前目录下运行样例代码需要PadddlePaddle Fluid的v.1.5或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据[安装文档](http://paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/install/index_cn.html)中的说明来更新PaddlePaddle。

### 数据准备

模型库中提供了download.py数据下载脚本，该脚本支持下载MNIST数据集，CycleGAN和Pix2Pix所需要的数据集。使用以下命令下载数据：
    python download.py --dataset=mnist
通过指定dataset参数来下载相应的数据集。

StarGAN, AttGAN和STGAN所需要的[Celeba](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)数据集可以自行下载。

**自定义数据集：**
用户可以使用自定义的数据集，只要设置成所对应的生成模型所需要的数据格式即可。

注意: pix2pix模型数据集准备中的list文件需要通过scripts文件夹里的make_pair_data.py来生成，可以使用以下命令来生成：
  python scripts/make_pair_data.py \
    --direction=A2B
用户可以通过设置`--direction`参数生成list文件，从而确保图像风格转变的方向。

### 模型训练
**下载预训练模型: **
本示例提供以下预训练模型:

[Pix2Pix的预训练模型]()

[CycleGAN的预训练模型]()

[StarGAN的预训练模型]()

[AttGAN的预训练模型]()

[STGAN的预训练模型]()

下载完预训练模型之后，通过设置infer.py中`--init_model`加载预训练模型，测试所需要的图片。
执行以下命令得到CyleGAN的预测结果：

  python infer.py \
    --model_net=cyclegan \
    --init_model=$(path_to_init_model) \
    --image_size=256 \
    --dataset_dir=$(path_to_data) \
    --input_style=$(A_or_B) \
    --net_G=$(generator_network) \
    --g_base_dims=$(base_dim_of_generator)

效果如图所示：


执行以下命令得到Pix2Pix的预测结果：

  python infer.py \
    --model_net=Pix2pix \
    --init_model=$(path_to_init_model) \
    --image_size=256 \
    --dataset_dir=$(path_to_data) \
    --net_G=$(generator_network)

效果如图所示：

执行以下命令得到StarGAN的预测结果：

  python infer.py \
    --model_net=StarGAN \
    --init_model=$(path_to_init_model)\
    --dataset_dir=$(path_to_data)

效果如图所示：

执行以下命令得到AttGAN的预测结果：

  python infer.py \
    --model_net=AttGAN \
    --init_model=$(path_to_init_model)\
    --dataset_dir=$(path_to_data)

效果如图所示：

执行以下命令得到STGAN的预测结果：

  python infer.py \
    --model_net=STGAN \
    --init_model=$(path_to_init_model)\
    --dataset_dir=$(path_to_data)

效果如图所示：

**开始训练：** 数据准备完毕后，可以通过一下方式启动训练：

  python train.py \
    --model_net=$(name_of_model) \
    --dataset=$(name_of_dataset) \
    --data_dir=$(path_to_data) \
    --train_list=$(path_to_train_data_list) \
    --test_list=$(path_to_test_data_list) \
    --batch_size=$(batch_size)

- 可选参数见：
  python train.py --help
- 每个GAN都给出了一份运行示例，放在scripts文件夹内。
- 用户可以通过设置model_net参数来选择想要训练的模型，通过设置dataset参数来选择训练所需要的数据集。

### 模型测试
模型测试是利用训练完成的生成模型进行图像生成。infer.py是主要的执行程序，调用示例如下：

  python infer.py \
    --model_net=$(name_of_model) \
    --init_model=$(path_to_model) \
    --dataset_dir=$(path_to_data)


## 参考文献
[1] [Goodfellow, Ian J.; Pouget-Abadie, Jean; Mirza, Mehdi; Xu, Bing; Warde-Farley, David; Ozair, Sherjil; Courville, Aaron; Bengio, Yoshua. Generative Adversarial Networks. 2014. arXiv:1406.2661 [stat.ML].](https://arxiv.org/abs/1406.2661)

[2] [https://zh.wikipedia.org/wiki/生成对抗网络](https://zh.wikipedia.org/wiki/生成对抗网络)

[3] [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)

[4] [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

[5] [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)

[6] [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

[7] [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)

[8] [AttGAN: Facial Attribute Editing by Only Changing What You Want](https://arxiv.org/abs/1711.10678)

[9] [STGAN: A Unified Selective Transfer Network for Arbitrary Image Attribute Editing](https://arxiv.org/abs/1904.09709)
