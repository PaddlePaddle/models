运行本目录下的范例模型需要安装PaddlePaddle 2.0-rc版。如果您的 PaddlePaddle 安装版本低于此要求，请按照[安装文档](https://www.paddlepaddle.org.cn/#quick-start)中的说明更新 PaddlePaddle 安装版本。

# Variational Autoencoder (VAE) for Text Generation
以下是本范例模型的简要目录结构及说明：

```text
.
├── README.md         # 文档
├── args.py           # 训练、预测以及模型参数配置程序
├── data.py           # 数据读入程序
├── download.py       # 数据下载程序
├── train.py          # 训练主程序
├── predict.py        # 预测主程序
└── model.py          # VAE模型组网部分，以及Metric等
```

## 简介
本目录下此范例模型的实现，旨在展示如何用Paddle 2.0-rc 构建用于文本生成的VAE示例，其中LSTM作为编码器和解码器。分别对官方PTB数据和yahoo数据集进行训练。

关于VAE的详细介绍参照： [(Bowman et al., 2015) Generating Sentences from a Continuous Space](https://arxiv.org/pdf/1511.06349.pdf)

## 数据介绍

本教程使用了两个文本数据集：

PTB dataset，原始下载地址为: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

yahoo，原始下载地址为：https://drive.google.com/file/d/13IsiffVjcQ-wrrbBGMwiG3sYf-DFxtXH/view?usp=sharing/

### 数据获取

```
python download.py --task ptb  # 下载ptb数据集

python download.py --task yahoo # 下载yahoo数据集

```

## 模型训练

如果使用ptb数据集训练，可以通过下面命令配置：

```
export CUDA_VISIBLE_DEVICES=0
python train.py \
        --batch_size 32 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset ptb \
        --model_path ptb_model\
        --use_gpu True \
        --max_epoch 50 \

```

如果需要多卡运行，可以运行如下命令：

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch train.py \
        --batch_size 32 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset ptb \
        --model_path ptb_model \
        --use_gpu True \
        --max_epoch 50 \

```

如果需要使用yahoo数据集进行多卡运行，可以将参数配置如下：

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch train.py \
        --batch_size 32 \
        --embed_dim 512 \
        --hidden_size 550 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset yahoo \
        --model_path yahoo_model \
        --use_gpu True \
        --max_epoch 50 \

```


**NOTE:** 如需恢复模型训练，则`init_from_ckpt`只需指定到文件名即可，不需要添加文件尾缀。如`--init_from_ckpt=ptb_model/49`即可，程序会自动加载模型参数`ptb_model/49.pdparams`，也会自动加载优化器状态`ptb_model/49.pdopt`。


## 模型预测

当模型训练完成之后，可以选择加载模型保存目录下的第 50 个epoch的模型进行预测，生成batch_size条短文本。如果使用ptb数据集，可以通过下面命令配置：

```
export CUDA_VISIBLE_DEVICES=0
python predict.py \
        --batch_size 32 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset ptb \
        --use_gpu True \
        --init_from_ckpt ptb_model/49 \

```

使用yahoo数据集，需要配置embed_dim和hidden_size：

```
python predict.py \
        --batch_size 32 \
        --init_scale 0.1 \
        --embed_dim 512 \
        --hidden_size 550 \
        --max_grad_norm 5.0 \
        --dataset yahoo \
        --use_gpu True \
        --init_from_ckpt yahoo_model/49 \

```

## 效果评价



||Test PPL|Test NLL|
|:-|:-:|:-:|
|ptb dataset|108.71|102.76|
|yahoo dataset|78.38|349.48|


## 生成样例

shareholders were spent about N shares to spend $ N million to ual sell this trust stock last week

new york stock exchange composite trading trading outnumbered closed at $ N a share down N cents

the company cited pressure to pursue up existing facilities in the third quarter was for <unk> and four N million briefly stocks for so-called unusual liability

people had <unk> down out the kind of and much why your relationship are anyway

there are a historic investment giant chips which ran the <unk> benefit the attempting to original maker
