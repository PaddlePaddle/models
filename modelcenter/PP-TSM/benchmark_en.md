## 1. Training Benchmark

### 1.1 Environment

* The training process of PP-TSM model uses 8 GPUs, every GPU batch size is 16 for training. If the number GPU and batch size of training do not use the above configuration, you shouldadjust the learning rate and number of iterations.

### 1.2 Datasets
The PP-TSM model uses Kinetics-400 dataset as the training and test set.

### 1.3 Benchmark

|Model name | task | input size | frames | ips |
|---|---|---|---|---|
|pptsm_k400_frames_uniform | action recognition | 224 | 8 | 274.32 |


## 2. Inference Benchmark

### 2.1 Environment

* The PP-TSM model's inference speed test is tested with single-card V100, batch size=1, CUDA 10.2, CUDNN 8.1.1, and TensorRT inference speed test using TensorRT 7.0.0.11.

### 2.2 Datasets

The PP-TSM model uses Kinetics-400 dataset as the training and test set.

### 2.3 Benchmark

| Model name | Accuracy% | Preprocess time/ms | Inference time/ms | Total inference time/ms |
| :---- | :----: | :----: |:----: |:----: |
|pptsm_k400_frames_uniform | 75.11 | 51.84 | 11.26 | 63.1 |


## 3. Reference
Ref: https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/benchmark.md
