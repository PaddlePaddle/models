# Angle closure Glaucoma Evaluation Challenge
The goal of the challenge is to evaluate and compare automated algorithms for angle closure classification and localization of scleral spur (SS) points on a common dataset of AS-OCT images. We invite the medical image analysis community to participate by developing and testing existing and novel automated classification and segmentation methods.
More detail [AGE challenge](https://age.grand-challenge.org/Details/).

## Angle closure classification task

1. Prepare data

	* We assume that you have downloaded data(two zip files), and stored @ `../datasets/`.
	* (Updated on August 5) Replace update files.
	* We provide a demo about `zip file extract`, `xlsx reader`, `data structure explore` and `Train/Val split`.

2. Train
	
	* We assume that you have downloaded data, extracted compressed files, and stored @ `../datasets/`.
	* Based on PaddlePaddle and [ResNet34](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/resnet.py), we provide a baseline about `pretrain weight download and load`, `datareader`, `computation graph of ResNet34 model`, `training` and `evaluation metrics`.

3. Inference

	* We assume that you have downloaded data, extracted compressed files, and stored @ `../datasets/`.
	* We assume that you store checkpoint files @ `../weights/`
	* Based on PaddlePaddle and [ResNet34](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/resnet.py), we provide a baseline about `inference` and `dump result to csv file`.
