
# DeepFM for CTR Prediction

## Introduction
This model implementation reproduces the result of the paper "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction" on Criteo dataset.

```text
@inproceedings{guo2017deepfm,
  title={DeepFM: A Factorization-Machine based Neural Network for CTR Prediction},
  author={Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li and Xiuqiang He},
  booktitle={the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI)},
  pages={1725--1731},
  year={2017}
}
```

## Environment
- **Now all models in PaddleRec require PaddlePaddle version 1.6 or higher, or suitable develop version.**

## Download and preprocess data

We evaluate the effectiveness of our implemented DeepFM on Criteo dataset. The dataset was used for the [Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/) hosted by Kaggle and includes 45 million users'click records. Each row is the features for an ad display and the first column is a label indicating whether this ad has been clicked or not. There are 13 continuous features and 26 categorical ones.

To preprocess the raw dataset, we min-max normalize continuous features to [0, 1] and filter categorical features that occur less than 10 times. The dataset is randomly splited into two parts: 90% is for training, while the rest 10% is for testing.

Download and preprocess data:
```bash
cd data && python download_preprocess.py && cd ..
```

After executing these commands, 3 folders "train_data", "test_data" and "aid_data" will be generated. The folder "train_data" contains 90% of the raw data, while the rest 10% is in "test_data". The folder "aid_data" contains a created feature dictionary "feat_dict.pkl2".

## Local Train

```bash
nohup python local_train.py --model_output_dir models >> train_log 2>&1 &
```

## Local Infer
```bash
nohup python infer.py --model_output_dir models --test_epoch 1 >> infer_log 2>&1 &
```
Note: The last log info is the total Logloss and AUC for all test data.

## Result
Reproducing this result requires training with default hyperparameters. The default hyperparameter is shown in `args.py`. Using the default hyperparameters (10 threads, 100 batch size, etc.), it takes about 1.8 hours for CPUs to iterate the training data for one round.

When the training set is iterated to the 22nd round, the testing Logloss is `0.44797` and the testing AUC is `0.8046`.
<p align="center">
<img src="./picture/deepfm_result.png" height=200 hspace='10'/> <br />
</p>

## Distributed Train
We emulate distributed training on a local machine. In default, we use 2 X 2，i.e. 2 pservers X 2 trainers。

**Note: we suggest to use Paddle >= 1.6 or [the latest Paddle](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/install/Tables.html#whl-dev) in distributed train.**

### Download and preprocess distributed demo dataset
This small demo dataset(a few lines from Criteo dataset) only test if distributed training can train.
```bash
cd dist_data && python dist_data_download.py && cd ..
```

### Distributed Train and Infer
Train
```bash
# 该sh不支持Windows
sh cluster_train.sh
```
params of cluster_train.sh：
- train_data_dir: path of train data
- model_output_dir: path of saved model
- is_local: local or distributed training(set 0 in distributed training)
- is_sparse: whether to use sparse update in embedding. If not set, default is flase.
- role: role of process(pserver or trainer)
- endpoints: ip:port of all pservers
- current_endpoint: ip:port of current pserver(role should be pserver)
- trainers: the number of trainers

other params explained in cluster_train.py

Infer
```bash
python infer.py --model_output_dir cluster_model --test_epoch 10 --num_feat 141443 --test_data_dir=dist_data/dist_test_data --feat_dict='dist_data/aid_data/feat_dict_10.pkl2'
```

Notes:
- **Proxy must be closed**, e.g. unset http_proxy, unset https_proxy.

- The first trainer(with trainer_id 0) saves model params.

- After each training, pserver processes should be stop manually. You can use command below:

>ps -ef | grep python

- We use Dataset API to load data，it's only supported on Linux now.

## Distributed Train with Fleet
Fleet is High-Level API for distributed training in PaddlePaddle. See [DeepFM example](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/deepFM) in Fleet Repo.
