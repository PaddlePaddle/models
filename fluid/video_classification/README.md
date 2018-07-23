# Video Classification Based on Temporal Segment Network
______________________________________________________________________________

Video classification has drawn a significant amount of attentions in the past few years. This page introduces how to perform video classification with PaddlePaddle Fluid, on the public UCF-101 dataset, based on the state-of-the-art Temporal Segment Network (TSN) method.

## Table of Contents
Installation

Data preparation

Training

Evaluation

Inference

Performance

### Installation
Running sample code in this directory requires PaddelPaddle Fluid v0.13.0 and later. If the PaddlePaddle on your device is lower than this version, please follow the instructions in <a href="http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html" rel="nofollow">installation document</a> and make an update.

### Data preparation

#### download UCF-101 dataset
Users can download the UCF-101 dataset by the provided link in <code>data/download_link</code>. After finish downloading, one can get a file named "UCF101.rar".

#### decode video into frame
To avoid the process of decoding videos in network training, we offline decode them into frames and save it in the <code>pickle</code> format, easily readable for python.

Users can refer to the script <code>data/video_decode.py</code> for video decoding.

#### split data into train and test
We follow the split 1 of UCF-101 dataset. After data splitting, users can get 9537 videos for training and 3783 videos for validation. The reference script is <code>data/split_data.py</code>

#### save pickle for training
As stated above, we save all data as <code>pickle</code> format for training. All information in each video is saved into one pickle, includes video id, frames binary and label. Please refer to the script <code>data/generate_train_data.py</code>.

#### generate train.list and test.list
There are two demo files <em>train.list</em> and <em>test.list</em> in <code>data/</code>, with each line seperated by SPACE.

