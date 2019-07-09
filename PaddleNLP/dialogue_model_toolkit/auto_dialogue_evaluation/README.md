# Auto Dialogue Evaluation
## 简介
### 任务说明
对话自动评估（Auto Dialogue Evaluation）评估开放领域对话系统的回复质量，能够帮助企业或个人快速评估对话系统的回复质量，减少人工评估成本。
1. 在无标注数据的情况下，利用负采样训练匹配模型作为评估工具，实现对多个对话系统回复质量排序；
2. 利用少量标注数据（特定对话系统或场景的人工打分），在匹配模型基础上进行微调，可以显著提高该对话系统或场景的评估效果。

### 效果说明
我们以四个不同的对话系统（seq2seq\_naive／seq2seq\_att／keywords／human）为例，使用对话自动评估工具进行自动评估。
1. 无标注数据情况下，直接使用预训练好的评估工具进行评估；
    在四个对话系统上，自动评估打分和人工评估打分spearman相关系数，如下：

    /|seq2seq\_naive|seq2seq\_att|keywords|human
    --|:--:|--:|:--:|--:
    cor|0.361|0.343|0.324|0.288

    对四个系统平均得分排序：

    人工评估|k(0.591)<n(0.847)<a(1.116)<h(1.240)
    --|--:
    自动评估|k(0.625)<n(0.909)<a(1.399)<h(1.683)

2. 利用少量标注数据微调后，自动评估打分和人工打分spearman相关系数，如下：

    /|seq2seq\_naive|seq2seq\_att|keywords|human
    --|:--:|--:|:--:|--:
    cor|0.474|0.477|0.443|0.378

## 快速开始
### 安装说明
1. paddle安装

    本项目依赖于Paddle Fluid 1.3.1 及以上版本，请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装
2. 安装代码

    克隆数据集代码库到本地
    ```
    git clone https://github.com/PaddlePaddle/models.git
    cd models/PaddleNLP/dialogue_model_toolkit/auto_dialogue_evaluation
    ```

3. 环境依赖

    python版本依赖python 2.7

### 开始第一次模型调用
1. 数据准备

    下载经过预处理的数据，运行该脚本之后，data目录下会存在unlabel_data(train.ids/val.ids/test.ids)，lable_data(四个任务数据train.ids/val.ids/test.ids)，以及word2ids.

    该项目只开源测试集数据，其他数据仅提供样例。
    ```
    sh download_data.sh
    ```
2. 模型下载

    我们开源了基于海量未标注数据训练好的模型，以及基于少量标注数据微调的模型，可供用户直接使用
    ```
    cd model_files
    sh download_model.sh
    ```

    我们提供了两种下载方式,以下载auto_dialogue_evaluation_matching_pretrained_model为例

    方式一：基于PaddleHub命令行工具（PaddleHub可参考[安装指南](https://github.com/PaddlePaddle/PaddleHub)进行安装)
    ```
    hub download auto_dialogue_evaluation_matching_pretrained_model --output_path ./
    ```

    方式二：直接下载
    ```
    wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/auto_dialogue_evaluation_matching_pretrained-1.0.0.tar.gz
    ```

3. 模型预测

    基于上面的模型和数据，可以运行下面的命令直接对对话数据进行打分(预测结果输出在test_path中).
    ```
    TASK=human
    python -u main.py \
      --do_infer True \
      --use_cuda \
      --test_path data/label_data/$TASK/test.ids \
      --init_model model_files/${TASK}_finetuned
    ```
4. 模型评估

    基于上面的模型和数据，可以运行下面的命令进行效果评估。

    评估预训练模型作为自动评估效果：
    ```
    for task in seq2seq_naive seq2seq_att keywords human
    do
    echo $task
    python -u main.py \
      --do_val True \
      --use_cuda \
      --test_path data/label_data/$task/test.ids \
      --init_model model_files/matching_pretrained \
      --loss_type L2
    done
    ```

    评估微调模型效果：
    ```
    for task in seq2seq_naive seq2seq_att keywords human
    do
      echo $task
      python -u main.py \
        --do_val True \
        --use_cuda \
        --test_path data/label_data/$task/test.ids \
        --init_model model_files/${task}_finetuned \
        --loss_type L2
    done
    ```

5. 训练与验证

    基于示例的数据集，可以运行下面的命令，进行第一阶段训练
    ```
    python -u main.py \
      --do_train True \
      --use_cuda \
      --save_path model_files_tmp/matching_pretrained \
      --train_path data/unlabel_data/train.ids \
      --val_path data/unlabel_data/val.ids
    ```

    在第一阶段训练基础上，可利用少量标注数据进行第二阶段训练
    ```
    TASK=human
    python -u main.py \
      --do_train True \
      --loss_type L2 \
      --use_cuda \
      --save_path model_files_tmp/${TASK}_finetuned \
      --init_model model_files/matching_pretrained \
      --train_path data/label_data/$TASK/train.ids \
      --val_path data/label_data/$TASK/val.ids \
      --print_step 1 \
      --save_step 1 \
      --num_scan_data 50
    ```

## 进阶使用
### 任务定义与建模
对话自动评估任务输入是文本对（上文，回复），输出是回复质量得分。
### 模型原理介绍
匹配任务（预测上下文是否匹配）和自动评估任务有天然的联系，该项目利用匹配任务作为自动评估的预训练；

利用少量标注数据，在匹配模型基础上微调。
### 数据格式说明
训练、预测、评估使用的数据示例如下，数据由三列组成，以制表符（'\t'）分隔，第一列是以空格分开的上文id，第二列是以空格分开的回复id，第三列是标签
```
723 236 7823 12 8     887 13 77 4       2
8474 13 44 34         2 87 91 23       0
```

注：本项目额外提供了分词预处理脚本（在preprocess目录下），可供用户使用，具体使用方法如下：
```
python tokenizer.py --test_data_dir ./test.txt.utf8 --batch_size 1 > test.txt.utf8.seg
```

### 代码结构说明
main.py：该项目的主函数，封装包括训练、预测、评估的部分

config.py：定义了该项目模型的相关配置，包括具体模型类别、以及模型的超参数

reader.py：定义了读入数据，加载词典的功能

evaluation.py：定义评估函数

init.py:定义模型load函数

run.sh：训练、预测、评估运行脚本

## 其他
如何贡献代码

如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。
