# Multi_task_learning 

## 1、Introduction
The pretraining is usually performed on corpus with restricted domains, it is expected that increasing the domain diversity by further pre-training on other corpus may improve the generalization capability. Hence, we incorporate masked language model and domain classify model by using corpus from various domains as an auxiliary tasks in the fine-tuning phase, along with MRC. Additionally, we explore multi-task learning by incorporating the supervised dataset from other NLP tasks to learn better language representation.

## 2、Quick Start
We use PaddlePaddle PALM(multi-task Learning Library) to train MRQA2019 MRC multi-task baseline model, download PALM:
```
git clone https://github.com/PaddlePaddle/PALM.git
```

### Environment
- Python >= 2.7
- cuda >= 9.0
- cudnn >= 7.0
- PaddlePaddle >= 1.5.0 Please refer to Installation Guide [Installation Guide](http://www.paddlepaddle.org/#quick-start)

### Data Preparation
#### Get data directly: 
User can get the data directly we provided: 
```
bash wget_data.sh
```

#### Convert MRC dataset to squad format data: 
To download the MRQA datasets, run
```
cd scripts && bash download_data.sh && cd ..
```
The training and prediction datasets will be saved in `./scripts/train/` and `./scripts/dev/`, respectively.

The Multi_task_learning model only supports dataset files in SQuAD format. Before running the model on MRQA datasets, one need to convert the official MRQA data to SQuAD format. To do the conversion, run
```
cd scripts && bash convert_mrqa2squad.sh && cd ..
```
The output files will be named as `xxx.raw.json`.

For convenience, we provide a script to combine all the training and development data into a single file respectively.
```
cd scripts && bash combine.sh && cd ..
```
The combined files will be saved in `./scripts/train/mrqa-combined.raw.json` and `./scripts/dev/mrqa-combined.raw.json`.

### Models Preparation
In this competition, We use google squad2.0 model as pretrain model [Model Link](https://worksheets.codalab.org/worksheets/0x3852e60a51d2444680606556d404c657)
we provide script to convert tensorflow model to paddle model
```
cd scripts && python convert_model_params.py  --init_tf_checkpoint tf_model --fluid_params_dir paddle_model && cd ..
```
or user can get the pretrain model and multi-task learning trained models we provided: 
```
bash wget_models.sh
```
## 3、Train and Predict
Preparing data, models, and task profiles for PALM
```
bash run_build_palm.sh
```

Start training: 
```
cd PALM
bash run_multi_task.sh
```

## 4、Evaluation
To evaluate the result, run
```
bash run_evaluation.sh
```
Note that we use the evaluation script for SQuAD 1.1 here, which is equivalent to the official one.

## 5、Performance
|  | dev in_domain(Macro-F1)| dev out_of_domain(Macro-F1) |
| ------------- | ------------ | ------------ |
| Official baseline | 77.87 | 58.67 |
| BERT | 82.40 | 66.35 |
| BERT + MLM | 83.19 | 67.45 |
| BERT + MLM + ParaRank | 83.51 | 66.83 |

BERT: reading comprehension single model.

BERT + MLM: reading comprehension single model as main task, mask language model as auxiliary task.

BERT + MLM + ParaRank: reading comprehension single model as main task, mask language model and paragraph classify rank as auxiliary tasks.

BERT config: configs/reading_comprehension.yaml 

MLM config: configs/mask_language_model.yaml

ParaRank config: configs/answer_matching.yaml

## Copyright and License
Copyright 2019 Baidu.com, Inc. All Rights Reserved Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and
limitations under the License.


