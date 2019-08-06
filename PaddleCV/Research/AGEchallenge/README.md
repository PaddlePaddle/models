# Angle closure Glaucoma Evaluation Challenge
The goal of the challenge is to evaluate and compare automated algorithms for angle closure classification and localization of scleral spur (SS) points on a common dataset of AS-OCT images. We invite the medical image analysis community to participate by developing and testing existing and novel automated classification and segmentation methods.
More detail [AGE challenge](https://age.grand-challenge.org/Details/).

## 1.Download data
After you sign up `Grand Challenge` and join the [AGE challenge](https://age.grand-challenge.org/Details/).

Dataset can be downloaded from the [Download page](https://age.grand-challenge.org/Download/)

We assume `Training100.zip` and `Validation_ASOCT_Image.zip` are stored @ `./AGE_challenge Baseline/datasets/`

## 2.Environment installation
* Python >= 3.5
* cuDNN >= 7.3
* CUDA 9
* paddlepaddle-gpu >= 1.5.0
* xlrd == 1.2.0
* tqdm == 4.32.2
* pycocotools == 2.0.0

More detail [PaddlePaddle Installation Manuals](https://www.paddlepaddle.org.cn/documentation/docs/en/1.5/beginners_guide/install/index_en.html)

## 3. Angle closure classification task

See `Classification/`.

## 4. Scleral spur localization task

We provide two baseline models for localization task.

See `LocalizationFCN/` and `LocalizationRCNN/`.