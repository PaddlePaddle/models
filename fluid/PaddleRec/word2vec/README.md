
# Skip-Gram Word2Vec Model

## Introduction


## Environment
You should install PaddlePaddle Fluid first.

## Dataset
The training data for the 1 Billion Word Language Model Benchmark的(http://www.statmt.org/lm-benchmark).

Download dataset:
```bash
cd data && ./download.sh && cd ..
```

## Model
This model implement a skip-gram model of word2vector.


## Data Preprocessing method

Preprocess the training data to generate a word dict.

```bash
python preprocess.py --data_path ./data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled --dict_path data/1-billion_dict
```

## Train
The command line options for training can be listed by `python train.py -h`.

### Local Train:
we set CPU_NUM=1 as default CPU_NUM to execute
```bash
export CPU_NUM=1 && \
python train.py \
        --train_data_path ./data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled \
        --dict_path data/1-billion_dict \
        --with_hs --with_nce --is_local \
        2>&1 | tee train.log
```


### Distributed Train
Run a 2 pserver 2 trainer distribute training on a single machine.
In distributed training setting, training data is splited by trainer_id, so that training data
 do not overlap among trainers

```bash
sh cluster_train.sh
```

## Infer

In infer.py we construct some test cases in the `build_test_case` method to evaluate the effect of word embeding:
We enter the test case (we are currently using the analogical-reasoning task: find the structure of A - B = C - D, for which we calculate A - B + D, find the nearest C by cosine distance, the calculation accuracy is removed Candidates for A, B, and D appear in the candidate) Then calculate the cosine similarity of the candidate and all words in the entire embeding, and print out the topK (K is determined by the parameter --rank_num, the default is 4).

Such as:
For: boy - girl + aunt = uncle
0 nearest aunt: 0.89
1 nearest uncle: 0.70
2 nearest grandmother: 0.67
3 nearest father:0.64

You can also add your own tests by mimicking the examples given in the `build_test_case` method.

To running test case from test files, please download the test files into 'test' directory
we provide test for each case with the following structure:
        `word1 word2 word3 word4`
so we can build it into `word1 - word2 + word3 = word4`

Forecast in training:

```bash
Python infer.py --infer_during_train 2>&1 | tee infer.log
```
Use a model for offline prediction:

```bash
Python infer.py --infer_once --model_output_dir ./models/[specific models file directory] 2>&1 | tee infer.log
```

## Train on Baidu Cloud
1. Please prepare some CPU machines on Baidu Cloud following the steps in [train_on_baidu_cloud](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/user_guides/howto/training/train_on_baidu_cloud_cn.rst)
1. Prepare dataset using preprocess.py.
1. Split the train.txt to trainer_num parts and put them on the machines.
1. Run training with the cluster train using the command in `Distributed Train` above.
