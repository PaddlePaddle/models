
# DNN for Click-Through Rate prediction

## Introduction
This model implements the DNN part proposed in the following paper:

```text
@inproceedings{guo2017deepfm,
  title={DeepFM: A Factorization-Machine based Neural Network for CTR Prediction},
  author={Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li and Xiuqiang He},
  booktitle={the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI)},
  pages={1725--1731},
  year={2017}
}
```

The DeepFm combines factorization machine and deep neural networks to model
both low order and high order feature interactions. For details of the
factorization machines, please refer to the paper [factorization
machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)

## Environment
You should install PaddlePaddle Fluid first, and run:

```shell
pip install -r requirements.txt
```

## Dataset
This example uses Criteo dataset which was used for the [Display Advertising
Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)
hosted by Kaggle.

Each row is the features for an ad display and the first column is a label
indicating whether this ad has been clicked or not. There are 39 features in
total. 13 features take integer values and the other 26 features are
categorical features. For the test dataset, the labels are omitted.

Download dataset:
```bash
cd data && ./download.sh && cd ..
```

## Model
This Demo only implement the DNN part of the model described in DeepFM paper.
DeepFM model will be provided in other model.


## Data Preprocessing method
To preprocess the raw dataset, the integer features are clipped then min-max
normalized to [0, 1] and the categorical features are one-hot encoded. The raw
training dataset are splited such that 90% are used for training and the other
10% are used for validation during training. In reader.py, training data is the first 
90% of data in train.txt, and validation data is the left.

## Train
The command line options for training can be listed by `python train.py -h`.

### Local Train:
```bash
python train.py \
        --train_data_path data/raw/train.txt \
        2>&1 | tee train.log
```

After training pass 1 batch 40000, the testing AUC is `0.801178` and the testing
cost is `0.445196`.

### Distributed Train
Run a 2 pserver 2 trainer distribute training on a single machine.
In distributed training setting, training data is splited by trainer_id, so that training data
 do not overlap among trainers

```bash
sh cluster_train.sh
```

## Infer
The command line options for infering can be listed by `python infer.py -h`.

To make inference for the test dataset:
```bash
python infer.py \
        --model_path models/ \
        --data_path data/raw/train.txt
```
Note: The AUC value in the last log info is the total AUC for all test dataset. Here, train.txt is splited inside the reader.py so that validation data does not have overlap with training data.

## Train on Baidu Cloud
1. Please prepare some CPU machines on Baidu Cloud following the steps in [train_on_baidu_cloud](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/user_guides/howto/training/train_on_baidu_cloud_cn.rst)
1. Prepare dataset using preprocess.py.
1. Split the train.txt to trainer_num parts and put them on the machines.
1. Run training with the cluster train using the command in `Distributed Train` above.

## Train on Paddle Cloud
If you want to run this training on PaddleCloud, you can use the script ```cloud.py```, you can change the arguments in ```trian.py``` through environments in PaddleCloud.