# Multi task learning 

## 1. Introduction
Multi task learning (MTL) has been used in many NLP tasks to obtain better language representations. Hence, we experiment with several auxiliary tasks to improve the generalization capability of a MRC model. The auxiliary tasks that we use include

 - Unsupervised Task: masked Language Model
 - Supervised Tasks:
   -  natural language inference
   -  paragraph ranking

In the MRQA 2019 shared task, We use [PALM](https://github.com/PaddlePaddle/PALM) v1.0 (a multi-task learning Library based on PaddlePaddle) to perform multi-task training, which makes the implementation of new tasks and pre-trained models much easier than from scratch.


## 2.Preparation

### Environment
- Python >= 2.7
- cuda >= 9.0
- cudnn >= 7.0
- PaddlePaddle 1.6 (Please refer to the Installation Guide [Installation Guide](http://www.paddlepaddle.org/#quick-start))
- PALM v1.0

### Install PALM
To install PALM v1.0, run the follwing command under `multi_task_learning/`,

```
git clone --branch v1.0 --depth 1 https://github.com/PaddlePaddle/PALM.git
```

For more instructions, see the PALM user guide: [README.md](https://github.com/PaddlePaddle/PALM/blob/v1.0/README.md)


### Dowload data 
 
To download the MRQA training and development data, as well as other auxiliary data for MTL, run

```
bash wget_data.sh
```
The downloaded data will be saved into `data/mrqa` (combined MRQA training and development data), `data/mrqa_dev` (seperated MRQA in-domain and out-of-domain data, for model evaluation), `mlm4mrqa` (training data for masked language model task) and `data/am4mrqa` (training data for paragraph matching task).

### Download pre-trained parameters 
In our MTL experiments, we use BERT as our shared encoder. The parameters are initialized from the Whole Word Masking BERT (BERTwwm), further fine-tuned on the SQuAD 2.0 task with synthetic generated question answering corpora. The model parameters in Tensorflow format can be downloaded [here](https://worksheets.codalab.org/worksheets/0x3852e60a51d2444680606556d404c657). The following command can be used to convert the parameters to the format that is readable for PaddlePaddle.

```
1、cd scripts
2、# download cased_model_01.tar.gz from link
3、mkdir cased_model_01 && mv cased_model_01.tar.gz cased_model_01 && cd cased_model_01 && tar -xvf cased_model_01.tar.gz && cd ..
4、python convert_model_params.py --init_tf_checkpoint cased_model_01/model.ckpt --fluid_params_dir params
5、mkdir squad2_model && mv cased_model_01/vocab.txt cased_model_01/bert_config.json params squad2_model 
```

Alternatively, user can directly **download the parameters that we have converted**: 

```
bash wget_pretrained_model.sh
```
## 3. Training
In the following example, we use PALM library to preform a MLT with 3 tasks (i.e. machine reading comprehension as main task, masked lagnuage model and paragraph ranking as auxiliary tasks). For a detialed instruction on PALM, please refer to the [user guide](https://github.com/PaddlePaddle/PALM/blob/v1.0/README.md).

The PALM library requires a config file for every single task and a main config file `mtl_config.yaml`, which control the training behavior and hyper-parameters. For simplicity, we have prepared those files in the `multi_task_learning/configs` folder. To move the configuration files, data set and model parameters to the correct directory, run

```
bash run_build_palm.sh
```

Once everything is in the right place, one can start training

```
cd PALM
bash run_multi_task.sh
```
The fine-tuned parameters and model predictions will be saved in `PALM/output/`, as specified by `mtl_config.yaml`.

## 4. Evaluation
The scripts for evaluation are in the folder `scripts/`. Here we provide an example for the usage of those scripts. 
Before evaluation, one need a json file which contains the prediction results on the MRQA dev set. For convenience, we prepare two model prediction files with different MTL configurations, which have been saved in the `prediction_results/` folder, as downloaded in section **Download data**. 

To evaluate the result, run

```
bash run_evaluation.sh
```
The F1 and EM score of the two model predictions will be saved into `prediction_results/BERT_MLM.log` and `prediction_results/BERT_MLM_ParaRank.log`. The macro average of F1 score will be printed on the console. The table below shows the results of our experiments with different MTL configurations.

|models |in-domain dev (Macro-F1)|out-of-domain dev (Macro-F1) |
| ------------- | ------------ | ------------ |
| Official baseline | 77.87 | 58.67 |
| BERT (no MTL) | 82.40 | 66.35 |
| BERT + MLM | 83.19 | 67.45 |
| BERT + MLM + ParaRank | 83.51 | 66.83 |


## Copyright and License
Copyright 2019 Baidu.com, Inc. All Rights Reserved Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and
limitations under the License.


