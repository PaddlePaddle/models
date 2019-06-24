
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
- Python 2.7

## Download and preprocess data

We evaluate the effectiveness of our implemented DeepFM on Criteo dataset. The dataset was used for the [Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/) hosted by Kaggle and includes 45 million users'click records. Each row is the features for an ad display and the first column is a label indicating whether this ad has been clicked or not. There are 13 continuous features and 26 categorical ones.

To preprocess the raw dataset, we min-max normalize continuous features to [0, 1] and  filter categorical features that occur less than 10 times. The dataset is randomly splited into two parts: 90% is for training, while the rest 10% is for testing.

Download and preprocess data:
```bash
cd data && sh download_preprocess.sh && cd ..
```

After executing these commands, two folders "raw_data" and "aid_data" will be generated. The folder "raw_data" contains raw data that is divided into several parts. The folder  "aid_data" contains a file "train_file_idx.pkl2" for data partitionin and a created feature dictionary "feat_dict.pkl2".

## Train

```bash
python local_train.py --model_output_dir models
```

## Infer
```bash
python infer.py --model_output_dir models --test_epoch 1
```
Note: The last log info is the total Logloss and AUC for all test data.

## Result
![avatar](./picture/deepfm_result.pdf)
