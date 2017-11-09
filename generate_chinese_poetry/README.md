# 中国古诗生成

## 简介
基于编码器-解码器(encoder-decoder)神经网络模型，利用全唐诗进行诗句-诗句(sequence to sequence)训练，实现给定诗句后，生成下一诗句。

以下是本例的简要目录结构及说明：

```text
.
├── data                 # 存储训练数据及字典
│   ├── download.sh      # 下载原始数据
├── models               # 存储训练好的模型
├── README.md            # 文档
├── index.html           # 文档(html格式)
├── preprocess.py        # 原始数据预处理
├── generate.py          # 生成诗句脚本
├── network_conf.py      # 模型定义
├── reader.py            # 数据读取接口
├── train.py             # 训练脚本
└── utils.py             # 定义实用工具函数
```

## 数据处理
### 原始数据来源
本例使用[中华古诗词数据库](https://github.com/chinese-poetry/chinese-poetry)中收集的全唐诗作为训练数据，共有约5.4万首唐诗。

### 原始数据下载
```bash
cd data && ./download.sh && cd ..
```
### 数据预处理
```bash
python preprocess.py --datadir data/raw --outfile data/poems.txt --dictfile data/dict.txt
```

上述脚本执行完后将生成处理好的训练数据poems.txt和数据字典dict.txt。poems.txt中每行为一首唐诗的信息，分为三列，分别为题目、作者、诗内容。
在诗内容中，诗句之间用'.'分隔。

训练数据示例：
```text
登鸛雀樓  王之渙  白日依山盡，黃河入海流.欲窮千里目，更上一層樓
觀獵     李白    太守耀清威，乘閑弄晚暉.江沙橫獵騎，山火遶行圍.箭逐雲鴻落，鷹隨月兔飛.不知白日暮，歡賞夜方歸
晦日重宴  陳嘉言  高門引冠蓋，下客抱支離.綺席珍羞滿，文場翰藻摛.蓂華彫上月，柳色藹春池.日斜歸戚里，連騎勒金羈
```

## 模型训练
训练脚本[train.py](./train.py)中的命令行参数如下：
```
Usage: train.py [OPTIONS]

Options:
  --num_passes INTEGER     Number of passes for the training task.
  --batch_size INTEGER     The number of training examples in one
                           forward/backward pass.
  --use_gpu TEXT           Whether to use gpu to train the model.
  --trainer_count INTEGER  The thread number used in training.
  --save_dir_path TEXT     The path to saved the trained models.
  --encoder_depth INTEGER  The number of stacked LSTM layers in encoder.
  --decoder_depth INTEGER  The number of stacked LSTM layers in decoder.
  --train_data_path TEXT   The path of trainning data.  [required]
  --word_dict_path TEXT    The path of word dictionary.  [required]
  --init_model_path TEXT   The path of a trained model used to initialized all
                           the model parameters.
  --help                   Show this message and exit.
```
### 参数说明
- `num_passes`: 训练pass数
- `batch_size`: batch大小
- `use_gpu`: 是否使用GPU
- `trainer_count`: trainer数目，默认为1
- `save_dir_path`: 模型存储路径，默认为当前目录下models目录
- `encoder_depth`: 模型中编码器LSTM深度，默认为3
- `decoder_depth`: 模型中解码器LSTM深度，默认为3
- `train_data_path`: 训练数据路径
- `word_dict_path`: 数据字典路径
- `init_model_path`: 初始模型路径，从头训练时无需指定

### 训练执行
```bash
python train.py \
    --num_passes 10 \
    --batch_size 256 \
    --use_gpu True \
    --trainer_count 1 \
    --save_dir_path models \
    --train_data_path data/poems.txt \
    --word_dict_path data/dict.txt \
    2>&1 | tee train.log
```
每个pass训练结束后，模型参数将保存在models目录下。训练日志保存在train.log中。

### 最优模型参数
寻找cost最小的pass，使用该pass对应的模型参数用于后续预测。
```bash
python -c 'import utils; utils.find_optiaml_pass("./train.log")'
```

## 生成诗句
使用[generate.py](./generate.py)脚本对输入诗句生成下一诗句，
命令行参数如下：
```
Usage: generate.py [OPTIONS]

Options:
  --model_path TEXT      The path of the trained model for generation.
  --word_dict_path TEXT  The path of word dictionary.  [required]
  --test_data_path TEXT  The path of input data for generation.  [required]
  --batch_size INTEGER   The number of testing examples in one forward pass in
                         generation.
  --beam_size INTEGER    The beam expansion in beam search.
  --save_file TEXT       The file path to save the generated results.
                         [required]
  --use_gpu TEXT         Whether to use GPU in generation.
  --help                 Show this message and exit.
```
### 参数说明
- `model_path`: 训练好的模型参数文件
- `word_dict_path`: 数据字典路径
- `test_data_path`: 输入数据路径
- `batch_size`: batch大小，默认为1
- `beam_size`: beam search中搜索范围大小，默认为5
- `save_file`: 输出保存路径
- `use_gpu`: 是否使用GPU

### 执行生成
例如将诗句 `白日依山盡，黃河入海流` 保存在文件 `input.txt` 中作为预测下句诗的输入，执行命令：
```bash
python generate.py \
    --model_path models/pass_00100.tar.gz \
    --word_dict_path data/dict.txt \
    --test_data_path input.txt \
    --save_file output.txt
```
生成结果将保存在文件 `output.txt` 中。
