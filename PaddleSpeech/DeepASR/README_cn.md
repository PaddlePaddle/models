[English](README.md)

运行本目录下的程序示例需要使用 PaddlePaddle v1.6.0及以上版本。如果您的 PaddlePaddle 安装版本低于此要求，请按照[安装文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/install/index_cn.html)中的说明更新 PaddlePaddle 安装版本。

---

DeepASR (Deep Automatic Speech Recognition) 是一个基于PaddlePaddle FLuid与[Kaldi](http://www.kaldi-asr.org)的语音识别系统。其利用Fluid框架完成语音识别中声学模型的配置和训练，并集成 Kaldi 的解码器。旨在方便已对 Kaldi 的较为熟悉的用户实现中声学模型的快速、大规模训练，并利用kaldi完成复杂的语音数据预处理和最终的解码过程。

### 目录
- [模型概览](#模型概览)
- [安装](#安装)
- [数据预处理](#数据预处理)
- [模型训练](#声学模型的训练)
- [训练过程中的时间分析](#训练过程中的时间分析)
- [预测和解码](#预测和解码)
- [错误率评估](#错误率评估)
- [Aishell 实例](#aishell-实例)
- [欢迎贡献更多的实例](#欢迎贡献更多的实例)

### 模型概览

DeepASR是一个单卷积层加多层层叠LSTMP 结构的声学模型，利用卷积来进行初步的特征提取，并用多层的LSTMP来对时序关系进行建模，使用交叉熵作为损失函数。[LSTMP](https://arxiv.org/abs/1402.1128)(LSTM with recurrent projection layer)是传统 LSTM 的拓展，在 LSTM 的基础上增加了一个映射层，将隐含层映射到较低的维度并输入下一个时间步，这种结构在大为减小 LSTM 的参数规模和计算复杂度的同时还提升了 LSTM 的性能表现。

<p align="center">
<img src="images/lstmp.png" height=240 width=480 hspace='10'/> <br />
图1 LSTMP 的拓扑结构
</p>

### 安装


#### kaldi的安装与设置


DeepASR解码过程中所用的解码器依赖于[Kaldi的安装](https://github.com/kaldi-asr/kaldi)，如环境中无Kaldi, 请`git clone`其源代码，并按给定的命令安装好kaldi，最后设置环境变量`KALDI_ROOT`：

```shell
export KALDI_ROOT=<kaldi的安装路径>

```
#### 解码器的安装
进入解码器源码所在的目录

```shell
cd models/fluid/DeepASR/decoder
```
运行安装脚本

```shell
sh setup.sh
```
 编译过程完成即成功地安转了解码器。

### 数据预处理

参考[Kaldi的数据准备流程](http://kaldi-asr.org/doc/data_prep.html)完成音频数据的特征提取和标签对齐

### 声学模型的训练

可以选择在CPU或GPU模式下进行声学模型的训练，例如在GPU模式下的训练

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py \
                   --train_feature_lst train_feature.lst \
                   --train_label_lst train_label.lst \
                   --val_feature_lst val_feature.lst \
                   --val_label_lst val_label.lst \
                   --mean_var global_mean_var \
                   --parallel
```
其中`train_feature.lst`和`train_label.lst`分别是训练数据集的特征列表文件和标注列表文件，类似的，`val_feature.lst`和`val_label.lst`对应的则是验证集的列表文件。实际训练过程中要正确指定 LSTMP 隐藏层的大小、学习率等重要参数。关于这些参数的说明，请运行

```shell
python train.py --help
```
获取更多信息。

### 训练过程中的时间分析

利用Fluid提供的性能分析工具profiler，可对训练过程进行性能分析，获取网络中operator级别的执行时间。

```shell
CUDA_VISIBLE_DEVICES=0 python -u tools/profile.py \
                   --train_feature_lst train_feature.lst \
                   --train_label_lst train_label.lst \
                   --val_feature_lst val_feature.lst \
                   --val_label_lst val_label.lst \
                   --mean_var global_mean_var
```


### 预测和解码

在充分训练好声学模型之后，利用训练过程中保存下来的模型checkpoint，可对输入的音频数据进行解码输出，得到声音到文字的识别结果.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u infer_by_ckpt.py \
                        --batch_size 96  \
                        --checkpoint deep_asr.pass_1.checkpoint \
                        --infer_feature_lst test_feature.lst  \
                        --infer_label_lst test_label.lst  \
                        --mean_var global_mean_var \
                        --parallel
```

### 错误率评估

对语音识别系统的评价常用的指标有词错误率(Word Error Rate, WER)和字错误率(Character Error Rate, CER), 在DeepASR中也实现了相关的度量工具，其运行方式为

```
python score_error_rate.py --error_rate_type cer --ref ref.txt --hyp decoding.txt
```
参数`error_rate_type`表示测量错误率的类型，即 WER 或 CER；`ref.txt` 和 `decoding.txt` 分别表示参考文本和实际解码出的文本，它们有着同样的格式：

```
key1 text1
key2 text2
key3 text3
...

```


### Aishell 实例

本节以[Aishell数据集](http://www.aishelltech.com/kysjcp)为例，展示如何完成从数据预处理到解码输出的过程。Aishell是由北京希尔贝克公司所开放的中文普通话语音数据集，时长178小时，包含了400名来自不同口音区域录制者的语音，原始数据可由[openslr](http://www.openslr.org/33)获取。为简化流程，这里提供了已完成预处理的数据集供下载：

```
cd examples/aishell
sh prepare_data.sh
```

其中包括了声学模型的训练数据以及解码过程中所用到的辅助文件等。下载数据完成后，在开始训练之前可对训练过程进行分析

```
sh profile.sh
```

执行训练

```
sh train.sh
```
默认是用4卡GPU进行训练，在实际过程中可根据可用GPU的数目和显存大小对`batch_size`、学习率等参数进行动态调整。训练过程中典型的损失函数和精度的变化趋势如图2所示

<p align="center">
<img src="images/learning_curve.png" height=480 width=640 hspace='10'/> <br />
图2 在Aishell数据集上训练声学模型的学习曲线
</p>

完成模型训练后，即可执行预测识别测试集语音中的文字：

```
sh infer_by_ckpt.sh
```

其中包括了声学模型的预测和解码器的解码输出两个重要的过程。以下是解码输出的样例：

```
...
BAC009S0764W0239 十一 五 期间 我 国 累计 境外 投资 七千亿 美元
BAC009S0765W0140 在 了解 送 方 的 资产 情况 与 需求 之后
BAC009S0915W0291 这 对 苹果 来说 不 是 件 容易 的 事 儿
BAC009S0769W0159 今年 土地 收入 预计 近 四万亿 元
BAC009S0907W0451 由 浦东 商店 作为 掩护
BAC009S0768W0128 土地 交易 可能 随着 供应 淡季 的 到来 而 降温
...
```

每行对应一个输出，均以音频样本的关键字开头，随后是按词分隔的解码出的中文文本。解码完成后运行脚本评估字错误率(CER)

```
sh score_cer.sh
```

其输出样例如下所示

```
Error rate[cer] = 0.101971 (10683/104765),
total 7176 sentences in hyp, 0 not presented in ref.
```

利用经过20轮左右训练的声学模型，可以在Aishell的测试集上得到CER约10%的识别结果。


### 欢迎贡献更多的实例

DeepASR目前只开放了Aishell实例，我们欢迎用户在更多的数据集上测试完整的训练流程并贡献到这个项目中。
