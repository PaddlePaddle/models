## 简介

### 任务说明
  机器翻译的输入一般是源语言的句子。但在很多实际系统中，比如语音识别系统的输出或者基于拼音的文字输入，源语言句子一般包含很多同音字错误, 这会导致翻译出现很多意想不到的错误。由于可以同时获得发音信息，我们提出了一种在输入端加入发音信息，进而在模型的嵌入层
融合文字信息和发音信息的翻译方法，大大提高了翻译模型对同音字错误的抵抗能力。

  文章地址：https://arxiv.org/abs/1810.06729

### 效果说明

  我们使用LDC Chinese-to-English数据集训练。中文词典用的是[DaCiDian](https://github.com/aishell-foundation/DaCiDian)。 在newstest2006上进行评测，效果如下所示：

| beta=0 | beta=0.50 | beta=0.85 | beta=0.95 |
|-|-|-|-|
| 47.96 | 48.71 | 48.85 | 48.46 |

beta代表发音信息的权重。这表明，即使将绝大部分权重放在发音信息上，翻译的效果依然很好。与此同时，翻译系统对同音字错误的抵抗力大大提高。


## 安装说明

1. paddle安装

   本项目依赖于 PaddlePaddle Fluid 1.3.1 及以上版本，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装

2. 环境依赖

   请参考PaddlePaddle[安装说明](http://paddlepaddle.org/documentation/docs/zh/1.3/beginners_guide/install/index_cn.html)部分的内容



## 如何训练

1. 数据格式

   数据格式和[Paddle机器翻译](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/neural_machine_translation/transformer)的格式一致。为了获得输入句子的发音信息，需要额外提供源语言的发音基本单元和发音的词典。

   A) 发音基本单元文件

   中文的发音基本单元是拼音，将所有的拼音放在一个文件，类似：

   <unk>

   bo

   li

   。。。

   B）发音词典

   根据DaCiDian，对bpe后的源语言中的token赋予一个或者几个发音，类似：

   ▁玻利维亚 bo li wei ya

   ▁举行 ju xing

   ▁总统 zong tong

   ▁与 yu

   巴斯 ba si

   ▁这个 zhei ge|zhe ge

   。。。

2. 训练模型

   数据准备完成后，可以使用 `train.py` 脚本进行训练。例子如下：

```sh
  python train.py \
  --src_vocab_fpath nist_data/vocab_all.28000 \
  --trg_vocab_fpath nist_data/vocab_all.28000 \
  --train_file_pattern nist_data/nist_train.txt \
  --phoneme_vocab_fpath nist_data/zh_pinyins.txt \
  --lexicon_fpath nist_data/zh_lexicon.txt \
  --batch_size 2048 \
  --use_token_batch True \
  --sort_type pool \
  --pool_size 200000 \
  --use_py_reader False \
  --use_mem_opt False \
  --enable_ce False \
  --fetch_steps 1 \
  pass_num 100 \
  learning_rate 2.0 \
  warmup_steps 8000 \
  beta2 0.997 \
  d_model 512 \
  d_inner_hid 2048 \
  n_head 8 \
  weight_sharing True \
  max_length 256 \
  save_freq 10000 \
  beta 0.85 \
  model_dir pinyin_models_beta085 \
  ckpt_dir pinyin_ckpts_beta085
```

上述命令中设置了源语言词典文件路径（`src_vocab_fpath`）、目标语言词典文件路径（`trg_vocab_fpath`）、训练数据文件（`train_file_pattern`，支持通配符), 发音单元文件路径（`phoneme_vocab_fpath`), 发音词典路径（`lexicon_fpath`）等数据相关的参数和构造 batch 方式（`use_token_batch` 指定了数据按照 token 数目或者 sequence 数目组成 batch）等 reader 相关的参数。有关这些参数更详细的信息可以通过执行以下命令查看：

```sh
python train.py --help
```

   更多模型训练相关的参数则在 `config.py` 中的 `ModelHyperParams` 和 `TrainTaskConfig` 内定义；`ModelHyperParams` 定义了 embedding 维度等模型超参数，`TrainTaskConfig` 定义了 warmup 步数等训练需要的参数。这些参数默认使用了 Transformer 论文中 base model 的配置，如需调整可以在该脚本中进行修改。另外这些参数同样可在执行训练脚本的命令行中设置，传入的配置会合并并覆盖 `config.py` 中的配置.

   注意，如训练时更改了模型配置，使用 `infer.py` 预测时需要使用对应相同的模型配置；另外，训练时默认使用所有 GPU，可以通过 `CUDA_VISIBLE_DEVICES` 环境变量来设置使用指定的 GPU。

## 如何预测

使用以上提供的数据和模型，可以按照以下代码进行预测，翻译结果将打印到标准输出:

```sh
python infer.py \
--src_vocab_fpath nist_data/vocab_all.28000 \
--trg_vocab_fpath nist_data/vocab_all.28000 \
--test_file_pattern nist_data/nist_test.txt \
--phoneme_vocab_fpath nist_data/zh_pinyins.txt \
--lexicon_fpath nist_data/zh_lexicon.txt \
--batch_size 32 \
model_path pinyin_models_beta085/iter_200000.infer.model \
beam_size 5 \
max_out_len 255 \
beta 0.85
```
