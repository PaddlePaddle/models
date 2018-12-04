
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
```bash
python train.py \
        --train_data_path data/enwik8 \
        --dict_path data/enwik8_dict \
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


## Train on Baidu Cloud
1. Please prepare some CPU machines on Baidu Cloud following the steps in [train_on_baidu_cloud](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/user_guides/howto/training/train_on_baidu_cloud_cn.rst)
1. Prepare dataset using preprocess.py.
1. Split the train.txt to trainer_num parts and put them on the machines.
1. Run training with the cluster train using the command in `Distributed Train` above.
