# knowledge_distillation

## 1、Introduction
Model ensemble can improve the generalization of MRC models. However, such approach is not efficient. Because the inference of an ensemble model is slow and a huge amount of resources are required. We leverage the technique of distillation to ensemble multiple models into a single model solves the problem of slow inference process.

## 2、Quick Start

### Environment
- Python >= 2.7
- cuda >= 9.0
- cudnn >= 7.0
- PaddlePaddle >= 1.5.0 Please refer to Installation Guide [Installation Guide](http://www.paddlepaddle.org/#quick-start)

### Data and Models Preparation
User can get the data and trained knowledge_distillation models directly we provided: 
```
bash wget_models_and_data.sh
```
user can get data and models directorys: 

data: 
./data/input/mlm_data: mask language model dataset
./data/input/mrqa_distill_data: mrqa dataset, it includes two parts: mrqa_distill.json(json data we calculate from teacher models), mrqa-combined.all_dev.raw.json(merge all mrqa dev dataset). 
./data/input/mrqa_evaluation_dataset: mrqa evaluation data(in_domain data and out_of_domain json data)

models: 
./data/pretrain_model/squad2_model: pretrain model(google squad2.0 model as pretrain model [Model Link](https://worksheets.codalab.org/worksheets/0x3852e60a51d2444680606556d404c657))
./saved_models/knowledge_distillation_model: baidu trained knowledge distillation model.

## 3、Train and Predict
Train and predict  knowledge distillation model
```
bash run_distill.sh
```

## 4、Evaluation
To evaluate the result, run
```
sh run_evaluation.sh
```
Note that we use the evaluation script for SQuAD 1.1 here, which is equivalent to the official one.

## 5、Performance

|  | dev in_domain(Macro-F1)| dev out_of_domain(Macro-F1) |
| ------------- | ------------ | ------------ |
| Official baseline | 77.87 | 58.67 |
| KD(4 teacher model-> student)| 83.67 | 67.34 |
KD: knowledge distillation model(ensemble 4 teacher models to student model)

## Copyright and License
Copyright 2019 Baidu.com, Inc. All Rights Reserved Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and
limitations under the License.


