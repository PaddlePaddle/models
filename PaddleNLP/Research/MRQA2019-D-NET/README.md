# D-NET

## Introduction
D-NET is the system Baidu submitted for MRQA (Machine Reading for Question Answering) 2019 Shared Task that focused on generalization of machine reading comprehension (MRC) models. Our system is built on a framework of pre-training and fine-tuning. The techniques of pre-trained language models, multi-task learning and knowledge distillation are employed to improve the generalization of MRC models and the experimental results show the effectiveness of these strategies. Our system is ranked at top 1 of all the participants in terms of averaged F1 score. Additionally, we won the first place for 10 of the 12 test sets and the second place for the other two in terms of F1 scores.

## Framework

<p align="center">
<img src="./images/D-NET_framework.png" width="500">
</p>

### D-NET includes 3 parts: 
#### multi_task_learning
We use PaddlePaddle PALM multi-task learning library [Link](https://github.com/PaddlePaddle/PALM) to train single model for  MRQA 2019 Shared Task.

#### knowledge_distillation 
Model ensemble can improve the generalization of MRC models, we leverage the technique of distillation to ensemble multiple models into a single  model, and no loss of accuracy, distillation solves the problem of slow inference process and reduce the use of a huge amount of resource.

#### server
MRQA2019 submission environment with baidu bert inference model and xlnet inference model.


## Copyright and License
Copyright 2019 Baidu.com, Inc. All Rights Reserved Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
