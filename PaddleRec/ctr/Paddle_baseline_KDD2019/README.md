# Paddle_baseline_KDD2019
Paddle baseline for KDD2019 "Context-Aware Multi-Modal Transportation Recommendation"(https://dianshi.baidu.com/competition/29/question)

This repository is the demo codes for the  KDD2019 "Context-Aware Multi-Modal Transportation Recommendation" competition using PaddlePaddle. It is written by python and uses PaddlePaddle to solve the task. Note that this repository is on developing and welcome everyone to contribute. The current baseline solution codes can get 0.68 - 0.69 score of online submission. As an example, my submission based on these networks programmed by PaddlePaddle is 0.6898.
The reason of the publication of this baseline codes is to encourage us to use PaddlePaddle and build the most powerful recommendation model via PaddlePaddle. 

The example codes are ran on Linux, python2.7, single machine with CPU . Note that distributed train options are not provided here, if you want to learn more about this, please check more modes examples on https://github.com/PaddlePaddle/models. About the speed of training, for one epoch, 1000 batch size, it would take about 8 mins to train the whole training instances generated from raw data using SGD optimizer (it would take relatively longer using Adam optimizer). 

The configuration and process of all the networks are fundamental, a lot of optimizations can be done based on them to achieve better results e.g. better cost function, more powerful feature engineering, designed model validation, NN optimization tricks...

The code is rough and from my daily use. They will be trimmed these days...
## Install PaddlePaddle
please visit the official site of PaddlePaddle(http://www.paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/install/index_cn.html) 
## preprocess feature
```python
python preprocess_dense.py # change for different feature strategy
python pre_test_dense.py 
```
preprocess.py and preprocess_dense.py is the code for preprocessing the raw data. Two versions are provided to deal with all sparse features and sparse plus dense features. Correspondingly, pre_process_test.py and pre_test_dense.py are the codes to preproccess test raw data. The training instances are saved in json. It is very easy to add new features. In our demo, all features are generated from provided raw data except for weather feature, which is gengerated from open weather records.
Note that the feature generated in this step need to fit in the input of the model input. Make sure we use the right version. In demo codes, The sparse plus dense features are used for network_confv6. 

## build the network
main network logic is in network_confv?.py. The networks are base on fm & deep related algorithms. I try several networks and public some of them. There may be some defects in the networks but all of them are functional. 

## train the network
```python
python local_train.py
```
In local_train.py and map_reader.py, I use dataset API, so we need to download the corresponding .whl package or clone codes on develop branch of PaddlePaddle. The reason to use this is the speed of feeding data is much faster.
Note that the input format feed into the network is self-defined. make sure we build the same format between training and test.  

## test results
```python
python generate_test.py
python build_submit.py
```
In generate_test.py and build_submit, for convenience, I use the whole train data to train the network and test the network with provided data without label



