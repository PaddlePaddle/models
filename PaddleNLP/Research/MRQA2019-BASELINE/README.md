# A PaddlePaddle Baseline for 2019 MRQA Shared Task

Machine Reading for Question Answering (MRQA), which requires machines to comprehend text and answer questions about it, is a crucial task in natural language processing.

Although recent systems achieve impressive results on the several benchmarks, these systems are primarily evaluated on in-domain accuracy. The [2019 MRQA Shared Task](https://mrqa.github.io/shared) focuses on testing the generalization  of the existing systems on out-of-domain datasets. 

In this repository, we provide a baseline for the 2019 MRQA Shared Task that is built on top of [PaddlePaddle](https://github.com/paddlepaddle/paddle), and it features:
* ***Pre-trained Language Model***: [ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE) (Enhanced Representation through kNowledge IntEgration) is a pre-trained language model that is designed to learn better language representations by incorporating linguistic knowledge masking. Our ERNIE-based baseline outperforms the MRQA official baseline that uses BERT by **6.1** point (marco-f1) on the out-of-domain dev set. 
* ***Multi-GPU Fine-tuning and Prediction***: Support for Multi-GPU fine-tuning and prediction to accelerate the experiments. 

You can use this repo as starter codebase for 2019 MRQA Shared Task and bootstrap your next model. 

## How to Run
### Environment Requirements
The MRQA baseline system has been tested on python2.7.13 and PaddlePaddle 1.5, CentOS 6.3.
The model is fine-tuned on 8 P40-GPUs, with batch size=4*8=32 in total.

### 1. Download Thirdparty Dependencies
We will use the evaluation script for *SQuAD v1.1*, which is equivelent to the official one for MRQA. To download the SQuAD v1.1 evaluation script, run
```
wget https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/ -O evaluate-v1.1.py
```

### 2. Download Dataset
To download the MRQA datasets, run

```
cd data && sh download_data.sh && cd ..
```
The training and prediction datasets will be saved in `./data/train/` and `./data/dev/`, respectively.

### 3. Preprocess
The baseline system only supports dataset files in SQuAD format. Before running the system on MRQA datasets, one need to convert the official MRQA data to SQuAD format. To do the conversion, run

```
cd data && sh convert_mrqa2squad.sh && cd ..
```
The output files will be named as `xxx.raw.json`.

For convenience, we provide a script to combine all the training and development data into a single file respectively

```
cd data && sh combine.sh && cd ..

```
The combined files will be saved in `./data/train/mrqa-combined.raw.json` and `./data/dev/mrqa-combined.raw.json`.


### 4. Fine-tuning with ERNIE
To get better performance than the official baseline, we provide a pretrained model - **ERNIE** for fine-tuning. To download the ERNIE parameters, run

```
sh download_pretrained_model.sh
```
The pretrained model parameters and config files will be saved in `./ernie_model`.

To start fine-tuning, run

```
sh run_finetuning.sh
```
The predicted results and model parameters will be saved in `./output`.

### 5. Prediction
Once fine-tuned, one can predict by specifying the model checkpoint file saved in `./output/` (E.g. step\_3000, step\_5000\_final)

```
sh run_predict.sh parameters_to_restore
```
Where `parameters_to_restore` is the model parameters used in the evaluatation (e.g. output/step\_5000\_final). The predicted results will be saved in `./output/prediction.json`. For convenience, we also provide **[fine-tuned model parameters](https://baidu-nlp.bj.bcebos.com/MRQA2019-PaddlePaddle-fine-tuned-model.tar.gz)** on MRQA datasets. The model is fine-tuned for 2 epochs on 8 P40-GPUs, with batch size=4*8=32 in total. The performerce is shown below,

##### in-domain dev  (F1/EM)

|      Model     | HotpotQA | NaturalQ | NewsQA | SearchQA | SQuAD | TriviaQA | Macro-F1 |
| :------------- | :---------: | :----------: | :---------: | :----------: | :---------: | :----------: |:----------: |
| baseline + EMA | 81.4/65.5 | 81.6/69.9 | 73.1/57.9 | 85.1/79.1 | 93.3/87.1 | 79.0/73.4 | 82.4 |
| baseline woEMA | 82.4/66.9 | 81.7/69.9 | 73.0/57.8 | 85.1/79.2 | 93.4/87.2 | 79.0/73.4 | 82.4 |

##### out-of-domain dev  (F1/EM)

|      Model     | BioASQ | DROP | DuoRC | RACE | RE | Textbook | Macro-F1 |
| :------------- | :---------: | :----------: | :---------: | :----------: | :---------: | :----------: |:----------: |
| baseline + EMA | 70.2/54.7 | 57.3/47.5 | 64.1/52.8 | 51.7/37.2 | 87.9/77.7 | 63.1/53.6 | 65.7 |
| baseline woEMA | 69.9/54.6 | 57.0/47.3 | 64.0/52.8 | 51.8/37.4 | 87.8/77.6 | 63.0/53.4 | 65.6 |

Note that we turn on exponential moving average (EMA) during training by default (in most cases EMA can improve performance) and save EMA parameters into the final checkpoint files. The predicted answers using EMA parameters are saved into `ema_predictions.json`.   


### 6. Evaluation
To evaluate the result, run

```
sh run_evaluation.sh
```
Note that we use the evaluation script for *SQuAD 1.1* here, which is equivalent to the official one.  

# Copyright and License
Copyright 2019 Baidu.com, Inc. All Rights Reserved
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
