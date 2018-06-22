DeepASR (Deep Automic Speech Recognition) 是一个基于PaddlePaddle FLuid与kaldi的语音识别系统。其利用PaddlePaddle Fluid完成声学模型的配置和训练，集成kaldi的解码器完，方便已对kaldi的较为熟悉的用户实现声学模型的快速、大规模训练，并利用kaldi完成复杂的解码过程。

###目录
- [安装](#installation)
- [模型训练](#training)
- [预测和解码](#infer-decoding)
- [Aishell示例](#aishell-example)
- [如何贡献更多的实例](#how-to-contrib)

### 安装

#### PaddlePaddle
运行DeepASR需要PadddlePaddle Fluid v0.13.0或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据[安装文档](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html)中的说明来更新PaddlePaddle。

####kaldi
DeepASR所用的解码器依赖于[kaldi](https://github.com/kaldi-asr/kaldi), 按其中的命令安装好kaldi后设置环境变量：

```shell
export KALDI_ROOT=<kaldi的安装路径>

```
####解码器的安装

```shell
   git clone https://github.com/PaddlePaddle/models.git
   cd models/fluid/DeepASR/decoder
   sh setup.sh
```
 完成解码器的编译和安装
 
###模型训练

###预测和解码

###Aishell示例

###如何贡献更多的实例
 

 

