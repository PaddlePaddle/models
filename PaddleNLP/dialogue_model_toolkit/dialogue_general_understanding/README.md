# 对话通用理解模块DGU
 - [一、简介](#一、简介)
 - [二、快速开始](#二、快速开始)
 - [三、进阶使用](#三、进阶使用)
 - [四、其他](#四、其他)

## 一、简介

### 任务说明

&ensp;&ensp;&ensp;&ensp;对话相关的任务中，Dialogue System常常需要根据场景的变化去解决多种多样的任务。任务的多样性（意图识别、槽位解析、DA识别、DST等等），以及领域训练数据的稀少，给Dialogue System的研究和应用带来了巨大的困难和挑战，要使得dialogue system得到更好的发展，需要开发一个通用的对话理解模型。为此，我们给出了基于BERT的对话通用理解模块(DGU: DialogueGeneralUnderstanding)，通过实验表明，使用base-model(BERT)并结合常见的学习范式，就可以在几乎全部对话理解任务上取得比肩甚至超越各个领域业内最好的模型的效果，展现了学习一个通用对话理解模型的巨大潜力。

### 效果说明

&ensp;&ensp;&ensp;&ensp;a、效果上，我们基于对话相关的业内公开数据集进行评测，效果如下表所示：

| task_name | udc | udc | udc | atis_slot | dstc2 | atis_intent | swda | mrda |
| :------ | :------ | :------ | :------ | :------| :------ | :------ | :------ | :------ |
| 对话任务 | 匹配 | 匹配 | 匹配 | 槽位解析 | DST | 意图识别 | DA | DA |
| 任务类型 | 分类 | 分类 | 分类 | 序列标注 | 多标签分类 | 分类 | 分类 | 分类 |
| 任务名称 | udc | udc | udc| atis_slot | dstc2 | atis_intent | swda | mrda |
| 评估指标 | R1@10 | R2@10 | R5@10 | F1 | JOINT ACC | ACC | ACC | ACC |
| SOTA | 76.70% | 87.40% | 96.90% | 96.89% | 74.50% | 98.32% | 81.30% | 91.70% |
| DGU | 82.02% | 90.43% | 97.75% | 97.10% | 89.57% | 97.65% | 80.19% | 91.43% |

&ensp;&ensp;&ensp;&ensp;b、数据集说明：

```
UDC: Ubuntu Corpus V1;
ATIS: 微软提供的公开数据集DSTC2，Airline Travel Information System;
DSTC2: 对话状态跟踪挑战（Dialog State Tracking Challenge）2;
MRDA: Meeting Recorder Dialogue Act;
SWDA：Switchboard Dialogue Act Corpus;
```

## 二、快速开始

### 1、安装说明

#### &ensp;&ensp;a、paddle安装

&ensp;&ensp;&ensp;&ensp;本项目依赖于PaddlePaddle 1.3.1 及以上版本，请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装

#### &ensp;&ensp;b、安装代码

&ensp;&ensp;&ensp;&ensp;克隆数据集代码库到本地

```
git clone https://github.com/PaddlePaddle/models.git
cd models/PaddleNLP/dialogue_model_toolkit/dialogue_general_understanding
```

#### &ensp;&ensp;c、环境依赖

&ensp;&ensp;&ensp;&ensp;python版本依赖python 2.7

### 2、开始第一次模型调用

#### &ensp;&ensp;a、数据准备（数据、模型下载，预处理）

&ensp;&ensp;&ensp;&ensp;i、数据下载

```
sh download_data.sh
```

&ensp;&ensp;&ensp;&ensp;ii、(非必需)下载的数据集中已提供了训练集，测试集和验证集，用户如果需要重新生成某数据集的训练数据，可执行：

```
cd dialogue_general_understanding/scripts && sh run_build_data.sh task_name
parameters：
task_name: udc, swda, mrda, atis, dstc2
```

#### &ensp;&ensp;b、模型下载

&ensp;&ensp;&ensp;&ensp;该项目中，我们基于BERT开发了相关的对话模型，对话模型训练时需要依赖BERT的模型做fine-tuning, 且提供了目前公开数据集上训练好的多个对话模型。

&ensp;&ensp;&ensp;&ensp;i、BERT pretrain模型下载：

```
sh download_pretrain_model.sh
```

&ensp;&ensp;&ensp;&ensp;ii、dialogue_general_understanding模块内对话相关模型下载：

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;方式一：基于PaddleHub命令行工具（PaddleHub安装方式 https://github.com/PaddlePaddle/PaddleHub)

```
hub download dmtk_models --output_path ./
tar -xvf dmtk_models_1.0.0.tar.gz
```

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;方式二：直接下载

```
sh download_models.sh
```

#### &ensp;&ensp;c、CPU、GPU训练设置

&ensp;&ensp;&ensp;&ensp;CPU训练和预测: 

```
请将run_train.sh和run_predict.sh内如下两行参数设置为: 
1、export CUDA_VISIBLE_DEVICES=
2、--use_cuda false
```

&ensp;&ensp;&ensp;&ensp;GPU训练和预测:  

```
请修改run_train.sh和run_predict.sh内如下两行参数设置为:
1、export CUDA_VISIBLE_DEVICES=4 (用户可自行指定空闲的卡)
2、--use_cuda true
```

#### &ensp;&ensp;d、训练 

&ensp;&ensp;&ensp;&ensp;方式一(推荐)：

```
sh run_train.sh task_name
parameters：
task_name: udc, swda, mrda, atis_intent, atis_slot, dstc2
```

&ensp;&ensp;&ensp;&ensp;方式二：

```
python -u train.py --task_name mrda \ # name model to use. [udc|swda|mrda|atis_intent|atis_slot|dstc2]

       --use_cuda true \      # If set, use GPU for training.
       --do_train true \        # Whether to perform training.
       --do_val true \         # Whether to perform evaluation on dev data set. 
       --do_test true \       # Whether to perform evaluation on test data set.
       --epoch 10 \          #  Number of epoches for fine-tuning.
       --batch_size 4096 \          # Total examples' number in batch for training. see also --in_tokens.
       --data_dir ./data/mrda \        # Path to training data.
       --bert_config_path ./uncased_L-12_H-768_A-12/bert_config.json \        # Path to the json file for bert model config.
       --vocab_path ./uncased_L-12_H-768_A-12/vocab.txt \          # Vocabulary path.
       --init_pretraining_params ./uncased_L-12_H-768_A-12/params \         # Init pre-training params which preforms fine-tuning from
       --checkpoints ./output/mrda \          # Path to save checkpoints.
       --save_steps 200 \            # The steps interval to save checkpoints.
       --learning_rate 2e-5 \           # Learning rate used to train with warmup.
       --weight_decay 0.01 \            # Weight decay rate for L2 regularizer.
       --max_seq_len 128 \            # Number of words of the longest seqence.
       --skip_steps 100 \             # The steps interval to print loss.
       --validation_steps 500 \           # The steps interval to evaluate model performance.
       --num_iteration_per_drop_scope 10 \         # The iteration intervals to clean up temporary variables. 
       --use_fp16 false         # If set, use fp16 for training.
```

#### &ensp;&ensp;e、预测 （推荐f的方式来进行预测评估）

&ensp;&ensp;&ensp;&ensp;方式一(推荐)：

```
sh run_predict.sh task_name
parameters：
task_name: udc, swda, mrda, atis_intent, atis_slot, dstc2
```

&ensp;&ensp;&ensp;&ensp;方式二：

```
python -u predict.py --task_name mrda \      # name model to use. [udc|swda|mrda|atis_intent|atis_slot|dstc2]
--use_cuda true \          # If set, use GPU for training.
--batch_size 4096 \         # Total examples' number in batch for training. see also --in_tokens.
--init_checkpoint ./output/mrda/step_6500 \         # Init model
--data_dir ./data/mrda \       # Path to training data.
--vocab_path ./uncased_L-12_H-768_A-12/vocab.txt \        # Vocabulary path.
--max_seq_len 128 \          # Number of words of the longest seqence.
--bert_config_path ./uncased_L-12_H-768_A-12/bert_config.json        # Path to the json file for bert model config.
```

#### &ensp;&ensp;f、预测+评估（推荐）

&ensp;&ensp;&ensp;&ensp;dialogue_general_understanding模块内提供已训练好的对话模型，可通过sh download_models.sh下载，用户如果不训练模型的时候，可使用提供模型进行预测评估：

```
sh run_eval_metrics.sh task_name
parameters：
task_name: udc, swda, mrda, atis_intent, atis_slot, dstc2
```

## 三、进阶使用

### 1、任务定义与建模

&ensp;&ensp;&ensp;&ensp;dialogue_general_understanding模块，针对数据集开发了相关的模型训练过程，支持分类，多标签分类，序列标注等任务，用户可针对自己的数据集，进行相关的模型定制；

### 2、模型原理介绍

&ensp;&ensp;&ensp;&ensp;本项目针对对话理解相关的问题，底层基于BERT，上层定义范式(分类，多标签分类，序列标注), 开源了一系列公开数据集相关的模型，供用户可配置地使用：

### 3、数据格式说明

&ensp;&ensp;&ensp;&ensp;训练、预测、评估使用的数据可以由用户根据实际的对话应用场景，自己组织数据。输入网络的数据格式统一为，示例如下：

```
[CLS] token11 token12 token13  [INNER_SEP] token11 token12 token13 [SEP]  token21 token22 token23 [SEP]  token31 token32 token33 [SEP]
```

&ensp;&ensp;&ensp;&ensp;输入数据以[CLS]开始，[SEP]分割内容为对话内容相关三部分，如上文，当前句，下文等，如[SEP]分割的每部分内部由多轮组成的话，使用[INNER_SEP]进行分割；第二部分和第三部分部分皆可缺省；

&ensp;&ensp;&ensp;&ensp;目前dialogue_general_understanding模块内已将数据准备部分集成到代码内，用户可根据上面输入数据格式，组装自己的数据；
### 4、代码结构说明

```
.
├── run_train.sh     				    # 训练执行脚本
├── run_predict.sh					# 预测执行脚本
├── run_eval_metrics.sh				# 评估执行脚本
├── download_data.sh				    # 下载数据脚本
├── download_models.sh				# 下载对话模型脚本
├── download_pretrain_model.sh		# 下载bert pretrain模型脚本
├── train.py						    # train流程
├── predict.py					    # predict流程
├── eval_metrics.py					# 指标评估
├── define_predict_pack.py            # 封装预测结果
├── finetune_args.py                  # 模型训练相关的配置参数
├── batching.py						# 封装yield batch数据
├── optimization.py 	                # 模型优化器
├── tokenization.py				    # tokenizer工具
├── reader/data_reader.py：			# 数据的处理和组装过程，每个数据集都定义一个类进行处理
├── README.md							# 文档
├── utils/*							# 定义了其他常用的功能函数
└── scripts							# 数据处理脚本集合
       ├── run_build_data.sh			# 数据处理运行脚本
       ├── build_atis_dataset.py		# 构建atis_intent和atis_slot训练数据
       ├── build_dstc2_dataset.py		# 构建dstc2训练数据
       ├── build_mrda_dataset.py		# 构建mrda训练数据
       ├── build_swda_dataset.py		# 构建swda训练数据
       ├── commonlib.py				    # 数据处理通用方法
       └── conf				            # 公开数据集中训练集、验证集、测试集划分
       
../../models/dialogue_model_toolkit/dialogue_general_understanding
├── bert.py 					        # 底层bert模型
├── define_paradigm.py				# 上层网络范式
└── create_model.py					# 创建底层bert模型+上层网络范式网络结构
```

### 5、如何组建自己的模型

&ensp;&ensp;&ensp;&ensp;用户可以根据自己的需求，组建自定义的模型，具体方法如下所示：

&ensp;&ensp;&ensp;&ensp;i、自定义数据 

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;如用户目前有数据集为**task_name**, 则在**data**下定义**task_name**文件夹，将数据集存放进去；在**reader/data_reader.py**中，新增自定义的数据处理的类，如**udc**数据集对应**UDCProcessor**;  在**train.py**内设置**task_name**和**processor**的对应关系(如**processors = {'udc': reader.UDCProcessor}**)，以及当前的数据集训练时是否是否使用**in_tokens**的方式计算batch大小(如：**in_tokens = {'udc': True}**)

&ensp;&ensp;&ensp;&ensp;ii、 自定义上层网络范式

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;如果用户自定义模型属于分类、多分类和序列标注这3种类型其中一个，则只需要在**paddle-nlp/models/dialogue_model_toolkit/dialogue_general_understanding/define_paradigm.py** 内指明**task_name**和相应上层范式函数的对应关系即可，如用户自定义模型属于其他模型，则需要自定义上层范式函数并指明其与**task_name**之间的关系；

&ensp;&ensp;&ensp;&ensp;iii、自定义预测封装接口

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;用户可在define_predict_pack.py内定义task_name和自定义封装预测接口的对应关系；

### 6、如何训练

&ensp;&ensp;&ensp;&ensp;i、按照上文所述的数据组织形式，组织自己的训练、评估、预测数据

&ensp;&ensp;&ensp;&ensp;ii、运行训练脚本

```
sh run_train.sh task_name
parameters：
task_name: 用户自定义名称
```

## 四、其他

### 如何贡献代码

&ensp;&ensp;&ensp;&ensp;如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。
