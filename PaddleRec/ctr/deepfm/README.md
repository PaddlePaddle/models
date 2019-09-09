
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
- PaddlePaddle 1.5

## Download and preprocess data

We evaluate the effectiveness of our implemented DeepFM on Criteo dataset. The dataset was used for the [Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/) hosted by Kaggle and includes 45 million users'click records. Each row is the features for an ad display and the first column is a label indicating whether this ad has been clicked or not. There are 13 continuous features and 26 categorical ones.

To preprocess the raw dataset, we min-max normalize continuous features to [0, 1] and filter categorical features that occur less than 10 times. The dataset is randomly splited into two parts: 90% is for training, while the rest 10% is for testing.

Download and preprocess data:
```bash
cd data && sh download_preprocess.sh && cd ..
```

After executing these commands, 3 folders "train_data", "test_data" and "aid_data" will be generated. The folder "train_data" contains 90% of the raw data, while the rest 10% is in "test_data". The folder "aid_data" contains a created feature dictionary "feat_dict.pkl2".

## Train

```bash
nohup python local_train.py --model_output_dir models >> train_log 2>&1 &
```

## Infer
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
