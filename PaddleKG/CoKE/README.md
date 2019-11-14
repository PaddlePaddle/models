# CoKE: Contextualized Knowledge Graph Embedding
## Introduction

This is the [PaddlePaddle](https://www.paddlepaddle.org.cn/) implementation of the [CoKE](https://arxiv.org/abs/1911.02168) model for Knowledge Graph Embedding(KGE).

CoKE is a novel KGE paradigm that learns dynamic, flexible, and fully contextualized entity and relation representations for a given Knowledge Graph(KG).
It takes a sequence of entities and relations as input, and uses [Transformer](https://arxiv.org/abs/1706.03762) to obtain contextualized representations for its components.
These representations are hence dynamically adaptive to the input, capturing contextual meanings of entities and relations therein.

Evaluation on a wide variety of public benchmarks verifies the superiority of CoKE in link prediction (also known as Knowledge Graph Completion, or KBC for short) and path query answering tasks.
CoKE performs consistently better than, or at least equally well as current state-of-the-art in almost every case.


## Requirements
The code has been tested running under the following environments:
- Python 3.6.5 with the following dependencies:
    -  PaddlePaddle 1.5.0
    -  numpy 1.16.3
- Python 2.7.14 for data_preprocess

- GPU environments:
    - CUDA 9.0, CuDNN v7 and NCCL 2.3.7
    - GPU: all the datasets run on 1 P40 GPU with our given configurations.  


## Model Training and Evaluation

### step1. Download dataset files
Download dataset files used in our paper by running:

```
sh wget_datasets.sh
```

This will first download the 4 widely used KBC datasets ([FB15k&WN18](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf),
[FB15k-237](https://www.aclweb.org/anthology/W15-4007/),
[WN18RR](https://arxiv.org/abs/1707.01476))
and 2 path query answering datasets ([wordnet_paths and freebase_paths](https://arxiv.org/abs/1506.01094)) .

Then it organize the train/valid/test files as the following `data` directory:

```
    data
    ├── fb15k
    │   ├── test.txt
    │   ├── train.txt
    │   └── valid.txt
    ├── fb15k237
    │   ├── test.txt
    │   ├── train.txt
    │   └── valid.txt
    ├── pathqueryFB  #the original data name is: freebase_paths
    │   ├── dev
    │   ├── test
    │   └── train
    ├── pathqueryWN  #the original data name is: wordnet_paths
    │   ├── dev
    │   ├── test
    │   └── train
    ├── wn18
    │   ├── test.txt
    │   ├── train.txt
    │   └── valid.txt
    └── wn18rr
        ├── test.txt
        ├── train.txt
        └── valid.txt
```

### step2. Data preprocess
Data preprocess commands are given in `data_preprocess.sh`.  
It takes raw train/valid/test files as input, and generates CoKE training and evaluation files.

```
sh data_preprocess.sh
```

### step3. Training

Model training commands are given in `kbc_train.sh` for KBC datasets, and `pathquery_train.sh` for pathquery datasets.
These scripts take a configuration file and GPU-ids as input arguments.
Train the model with a given configuration file.

For example, the following commands train *fb15k* and *pathqueryFB* each with a configuration file:

```
sh kbc_train.sh ./configs/fb15k_job_config.sh 0
sh pathquery_train.sh ./configs/pathqueryFB_job_config.sh 0
```


### step4. Evaluation
Model evaluation commands are given in `kbc_test.sh` for KBC datasets, and `pathquery_test.sh` for pathquery datasets.
These scripts take a configuration file and GPU-ids as input arguments.

For example, the following commands evaluate on *fb15k* and *pathqueryFB*:

```
sh kbc_test.sh ./configs/fb15k_job_config.sh 0
sh pathquery_test.sh ./configs/pathqueryFB_job_config.sh 0
```

We also provide trained model checkpoints on the 4 KBC datasets. Download these models to `kbc_models` directory using the following command:


```
sh wget_kbc_models.sh
```

The `kbc_models` contains the following files:

```
kbc_models
├── fb15k
│   ├── models
│   └── vocab.txt  #md5: 0720db5edbda69e00c05441a615db152
├── fb15k237
│   ├── models
│   └── vocab.txt  #md5: e843936790e48b3cbb35aa387d0d0fe5
├── wn18
│   ├── models
│   └── vocab.txt  #md5: 4904a9300fc3e54aea026ecba7d2c78e
└── wn18rr
    ├── models
    └── vocab.txt  #md5: c76aecebf5fc682f0e7922aeba380dd6
```

Check that your preprocessed `vocab.txt` files are identical to ours before evaluation with these models.


## Results
Results on KBC datasets:

|Dataset | MRR | HITS@1 |  HITS@5 | HITS@10 |
|---|---|---|---|---|
|FB15K | 0.852 | 0.823 |0.868 |    0.904 |
|FB15K237| 0.361 | 0.269 | 0.398 | 0.547 |
|WN18| 0.951 | 0.947 |0.954 | 0.960|
|WN18RR| 0.475 | 0.437 | 0.490 | 0.552 |

Results on path query datasets:

|Dataset | MQ | HITS@10 |
|---|---|---|
|Freebase | 0.948  | 0.764|
|WordNet |0.942 | 0.674 |

## Reproducing the results

Here are the configs to reproduce our results.
These are also given in the `configs/${TASK}_job_config.sh` files.

| Dataset | NetConfig | lr | softlabel | epoch | batch_size | dropout |
|---|---|---|---|---|---| ---|
|FB15K| L=6, H=256, A=4| 5e-4 | 0.8 | 300 | 512| 0.1 |
|WN18| L=6, H=256, A=4| 5e-4| 0.2 | 500 | 512 | 0.1 |
|FB15K237| L=6, H=256, A=4| 5e-4| 0.25 | 800 | 512 | 0.5 |
|WN18RR| L=6, H=256, A=4|3e-4 | 0.15 | 800 | 1024 | 0.1 |
|pathqueryFB | L=6, H=256, A=4 | 3e-4 | 1 | 10 | 2048 | 0.1 |
|pathqueryWN | L=6, H=256, A=4 | 3e-4 | 1 | 5 | 2048 | 0.1 |

## Citation
If you use any source code included in this project in your work, please cite the following paper:

```
@article{wang2019:coke,
  title={CoKE: Contextualized Knowledge Graph Embedding},
  author={Wang, Quan and Huang, Pingping and Wang, Haifeng and Dai, Songtai and Jiang, Wenbin and Liu, Jing and Lyu, Yajuan and Wu, Hua},
  journal={arXiv:1911.02168},
  year={2019}
}
```


## Copyright and License
Copyright 2019 Baidu.com, Inc. All Rights Reserved Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
