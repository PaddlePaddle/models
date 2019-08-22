# KT-NET

## Introduction

KT-NET (Knowledge and Text fusion NET) is a machine reading comprehension (MRC) model which integrates knowledge from knowledge bases (KBs) into pre-trained contextualized representations. The model is proposed in ACL2019 paper [Enhancing Pre-Trained Language Representations with Rich Knowledge for Machine Reading Comprehension](https://www.aclweb.org/anthology/P19-1226). The overall architecture of the model is shown as follows:

<p align="center">
<img src="images/architecture.png" width = "340" height = "300" /> <br />
Overall Architecture of KT-NET
</p>

This repository contains the PaddlePaddle implementation of KT-NET. The trained checkpoints are also provided for reproducing the result in the paper.

## How to Run

### Environment

This project should work fine if the following requirements have been satisfied:
+ python >= 3.7
+ paddlepaddle-gpu >= 1.5.1
+ NLTK >= 3.3
+ tqdm
+ CUDA, CuDNN and NCCL (CUDA 9.0, CuDNN v7 and NCCL 2.3.7 are recommended)

All of the experiments in the paper are performed on 4 P40 GPUs.

### Download the MRC datasets

In this work, we empirically evaluate our model on two benchmarks: 

#### 1. ReCoRD

[ReCoRD](https://sheng-z.github.io/ReCoRD-explorer/) (Reading Comprehension with Commonsense Reasoning Dataset) is a large-scale MRC dataset requiring commonsense reasoning. The official dataset in JSON format can be downloaded using Google drive (training set: [link](https://drive.google.com/file/d/1PoHmphyH79pETNws8kU2OwuerU7SWLHj/view), valid set: [link](https://drive.google.com/file/d/1WNaxBpXEGgPbymTzyN249P4ub-uU5dkO/view)). Please place the downloaded files `train.json` and `dev.json` into the `data/ReCoRD/` directory of this repository. We will also use the official evaluation script of ReCoRD, please run the following command at `KTNET/src/eval/` directory:
```
wget https://sheng-z.github.io/ReCoRD-explorer/evaluation.py -O record_official_evaluate.py
```

#### 2. SQuAD v1.1

[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/).

### Retrieve KB entries

Relevant knowledge should be retrieved and encoded before training the model. In this project, we leveraged two KBs: [WordNet](https://wordnet.princeton.edu/) and [NELL](http://rtw.ml.cmu.edu/rtw/). WordNet records lexical relations between words and NELL stores beliefs about entities. The following procedure describes how we retrieve relevant WordNet synsets and NELL concepts on MRC samples.

#### 1. Tokenization

