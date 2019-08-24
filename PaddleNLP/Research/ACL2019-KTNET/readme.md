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
+ paddlepaddle-gpu (the latest develop version is recommended)
+ NLTK >= 3.3
+ tqdm
+ CoreNLP (3.8.0 version is recommended)
+ pycorenlp
+ CUDA, CuDNN and NCCL (CUDA 9.0, CuDNN v7 and NCCL 2.3.7 are recommended)

All of the experiments in the paper are performed on 4 P40 GPUs.

### Download the MRC datasets

In this work, we empirically evaluate our model on two benchmarks: 

#### 1. ReCoRD

[ReCoRD](https://sheng-z.github.io/ReCoRD-explorer/) (Reading Comprehension with Commonsense Reasoning Dataset) is a large-scale MRC dataset requiring commonsense reasoning. The official dataset in JSON format can be downloaded using Google drive (training set: [link](https://drive.google.com/file/d/1PoHmphyH79pETNws8kU2OwuerU7SWLHj/view), valid set: [link](https://drive.google.com/file/d/1WNaxBpXEGgPbymTzyN249P4ub-uU5dkO/view)). Please place the downloaded files `train.json` and `dev.json` into the `data/ReCoRD/` directory of this repository. We will also use the official evaluation script of ReCoRD, so please run the following command:
```
wget https://sheng-z.github.io/ReCoRD-explorer/evaluation.py -O record_official_evaluate.py
mv record_official_evaluate.py KTNET/src/eval/
```

#### 2. SQuAD v1.1

[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) is a well-known extractive MRC dataset that consists of questions created by crowdworkers for Wikipedia articles. Please run the following command to download the official dataset and evaluation script.
```
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
mv train-v1.1.json dev-v1.1.json data/SQuAD/
wget https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/ -O squad_v1_official_evaluate.py
mv squad_v1_official_evaluate.py KTNET/src/eval/
```

### Retrieve KB entries

Relevant knowledge should be retrieved and encoded before training the model. In this project, we leveraged two KBs: [WordNet](https://wordnet.princeton.edu/) and [NELL](http://rtw.ml.cmu.edu/rtw/). WordNet records lexical relations between words and NELL stores beliefs about entities. The following procedure describes how we retrieve relevant WordNet synsets and NELL concepts for MRC samples.

#### 1. Named entity recognition (only for SQuAD)

To retrieve NELL concepts about entities, the named entity mentions in MRC samples should be annotated. For ReCoRD, the entity mentions have been provided in the dataset. For SQuAD, named entity recognition (NER) needs to be performed before the retrieval. We use [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html) in this step. After CoreNLP is [downloaded](http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip) and unzipped, run the following command at the CoreNLP directory to start the CoreNLP server:
```
java -mx10g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```
Then run the command:
```
cd retrieve_concepts/ner_tagging_squad
python3 tagging.py
```
The tagged dataset will be saved at `retrieve_concepts/ner_tagging_squad/output` directory. We have provided our output files for convenience ([download link](TODO)).

#### 2. Tokenization

Tokenization should be performed for retrieval. We use the same tokenizer with [BERT](https://github.com/google-research/bert). 
For ReCoRD, run the following command (or directly download our output from [link](TODO)):
```
cd retrieve_concepts/tokenization_record
python3 do_tokenization.py --do_lower_case
```
For SQuAD, run the following command (or directly download our output from [link](TODO)):
```
cd retrieve_concepts/tokenization_squad
python3 do_tokenization.py --do_lower_case
```

