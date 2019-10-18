# Text matching on Quora qestion-answer pair dataset

## contents

* [Introduction](#introduction)
  * [a brief review of the Quora Question Pair (QQP) Task](#a-brief-review-of-the-quora-question-pair-qqp-task)
  * [Our Work](#our-work)
* [Environment Preparation](#environment-preparation)
  * [Install Fluid release 1.0](#install-fluid-release-10)
    * [cpu version](#cpu-version)
    * [gpu version](#gpu-version)
    * [Have I installed Fluid successfully?](#have-i-installed-fluid-successfully)
* [Prepare Data](#prepare-data)
* [Train and evaluate](#train-and-evaluate)
* [Models](#models)
* [Results](#results)


## Introduction

### a brief review of the Quora Question Pair (QQP) Task

The [Quora Question Pair](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) dataset contains 400,000 question pairs from [Quora](https://www.quora.com/), where people ask and answer questions related to specific areas. Each sample in the dataset consists of two questions (both English) and a label that represents whether the questions are duplicate. The dataset is well annotated by human.

Below are two samples from the dataset. The last column indicates whether the two questions are duplicate (1) or not (0).

|id | qid1 | qid2| question1| question2| is_duplicate
|:---:|:---:|:---:|:---:|:---:|:---:|
|0 |1 |2 |What is the step by step guide to invest in share market in india? |What is the step by step guide to invest in share market? |0|
|1 |3 |4 |What is the story of Kohinoor (Koh-i-Noor) Diamond? | What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back? |0|

 A [kaggle competition](https://www.kaggle.com/c/quora-question-pairs#description) was held based on this dataset in 2017. The kagglers were given a training dataset (with labels), and requested to make predictions on a test dataset (without labels). The predictions were evaluated by the log-likelihood loss on the test data.

The kaggle competition has inspired much effective work. However, most of these models are rule-based and difficult to be transferred to new tasks. Researchers are seeking for more general models that work well on this task and other natual language processing (NLP) tasks.

[Wang _et al._](https://arxiv.org/abs/1702.03814) proposed a bilateral multi-perspective matching (BIMPM) model based on the Quora Question Pair dataset. They splitted the original dataset to [3 parts](https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing): _train.tsv_ (384,348 samples), _dev.tsv_ (10,000 samples) and _test.tsv_ (10,000 samples). The class distribution of _train.tsv_ is unbalanced (37% positive and 63% negative), while those of _dev.tsv_ and _test.tsv_ are balanced(50% positive and 50% negetive). We used the same splitting method in our experiments.

### Our Work

Based on the Quora Question Pair Dataset, we implemented some classic models in the area of neural language understanding (NLU). The accuracy of prediction results are evaluated on the _test.tsv_ from [Wang _et al._](https://arxiv.org/abs/1702.03814).

## Environment Preparation

### Install Fluid release 1.0

Please follow the [official document in English](http://www.paddlepaddle.org/documentation/docs/en/1.0/build_and_install/pip_install_en.html) or [official document in Chinese](http://www.paddlepaddle.org/documentation/docs/zh/1.0/beginners_guide/install/Start.html) to install the Fluid deep learning framework.

#### Have I installed Fluid successfully?

Run the following script from your command line:

```shell
python -c "import paddle"
```

If Fluid is installed successfully you should see no error message. Feel free to open issues under the [PaddlePaddle repository](https://github.com/PaddlePaddle/Paddle/issues) for support.

## Prepare Data

Please download the Quora dataset from [Google drive](https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing) and unzip to $HOME/.cache/paddle/dataset.

Then run _data/prepare_quora_data.sh_ to download the pre-trained _word2vec_ embedding file -- _glove.840B.300d.zip_:

```shell
sh data/prepare_quora_data.sh  
```

At this point the dataset directory ($HOME/.cache/paddle/dataset) structure should be:

```shell

$HOME/.cache/paddle/dataset
    |- Quora_question_pair_partition
        |- train.tsv
        |- test.tsv
        |- dev.tsv
        |- readme.txt
        |- wordvec.txt
    |- glove.840B.300d.txt
```

## Train and evaluate

We provide multiple models and configurations. Details are shown in `models` and `configs` directories. For a quick start, please run the _cdssmNet_ model with the corresponding configuration:

```shell
python train_and_evaluate.py  \
    --model_name=cdssmNet  \
    --config=cdssm_base
```

Logs will be output to the console. If everything works well, the logging information will have the same formats as the content in _cdssm_base.log_.

All configurations used in our experiments are as follows:

|Model|Config|command
|:----:|:----:|:----:|
|cdssmNet|cdssm_base|python train_and_evaluate.py  --model_name=cdssmNet  --config=cdssm_base
|DecAttNet|decatt_glove|python train_and_evaluate.py --model_name=DecAttNet  --config=decatt_glove
|InferSentNet|infer_sent_v1|python train_and_evaluate.py --model_name=InferSentNet --config=infer_sent_v1
|InferSentNet|infer_sent_v2|python train_and_evaluate.py --model_name=InferSentNet --config=infer_sent_v2
|SSENet|sse_base|python train_and_evaluate.py  --model_name=SSENet  --config=sse_base

## Models

We implemeted 4 models for now: the convolutional deep-structured semantic model (CDSSM, CNN-based), the InferSent model (RNN-based), the shortcut-stacked encoder (SSE, RNN-based), and the decomposed attention model (DecAtt, attention-based).

|Model|features|Context Encoder|Match Layer|Classification Layer
|:----:|:----:|:----:|:----:|:----:|
|CDSSM|word|1 layer conv1d|concatenation|MLP
|DecAtt|word|Attention|concatenation|MLP
|InferSent|word|1 layer Bi-LSTM|concatenation/element-wise product/<br>absolute element-wise difference|MLP
|SSE|word|3 layer Bi-LSTM|concatenation/element-wise product/<br>absolute element-wise difference|MLP

### CDSSM

```
@inproceedings{shen2014learning,
  title={Learning semantic representations using convolutional neural networks for web search},
  author={Shen, Yelong and He, Xiaodong and Gao, Jianfeng and Deng, Li and Mesnil, Gr{\'e}goire},
  booktitle={Proceedings of the 23rd International Conference on World Wide Web},
  pages={373--374},
  year={2014},
  organization={ACM}
}
```

### InferSent

```
@article{conneau2017supervised,
  title={Supervised learning of universal sentence representations from natural language inference data},
  author={Conneau, Alexis and Kiela, Douwe and Schwenk, Holger and Barrault, Loic and Bordes, Antoine},
  journal={arXiv preprint arXiv:1705.02364},
  year={2017}
}
```

### SSE

```
@article{nie2017shortcut,
  title={Shortcut-stacked sentence encoders for multi-domain inference},
  author={Nie, Yixin and Bansal, Mohit},
  journal={arXiv preprint arXiv:1708.02312},
  year={2017}
}
```

### DecAtt

```
@article{tomar2017neural,
  title={Neural paraphrase identification of questions with noisy pretraining},
  author={Tomar, Gaurav Singh and Duque, Thyago and T{\"a}ckstr{\"o}m, Oscar and Uszkoreit, Jakob and Das, Dipanjan},
  journal={arXiv preprint arXiv:1704.04565},
  year={2017}
}
```

## Results

|Model|Config|dev accuracy| test accuracy
|:----:|:----:|:----:|:----:|
|cdssmNet|cdssm_base|83.56%|82.83%|
|DecAttNet|decatt_glove|86.31%|86.22%|
|InferSentNet|infer_sent_v1|87.15%|86.62%|
|InferSentNet|infer_sent_v2|88.55%|88.43%|
|SSENet|sse_base|88.35%|88.25%|

In our experiment, we found that LSTM-based models outperformed convolution-based models. The DecAtt model has fewer parameters than LSTM-based models, but is sensitive to hyper-parameters.

<p align="center">

 <img src="imgs/models_test_acc.png" width = "500" alt="test_acc"/>

</p>
