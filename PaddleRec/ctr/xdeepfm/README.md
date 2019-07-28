
# xDeepFM for CTR Prediction

## Introduction
Reproduce [the open source code](https://github.com/Leavingseason/xDeepFM) of the paper "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems" with PaddlePaddle on demo data.

## Environment
- PaddlePaddle 1.5

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
When the training set is iterated to the 10th round, the testing Logloss is `0.48657` and the testing AUC is `0.7308`.
