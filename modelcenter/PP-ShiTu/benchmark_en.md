## 1. Train Benchmark

### 1.1 Software and hardware environment

* The feature extraction model of PP-ShiTu uses 8 GPUs in the training process, the batch size of each GPU is 256 for training. If the number of training GPUs and batch size are not consistent with the above configuration, you must refer to the FAQ to adjust the learning rate and the number of iterations.

* 8 GPUs are used in the training process of the detection model of PP-ShiTu, and the batch size of each GPU is 28 for training. If the number of training GPUs and batch size are not consistent with the above configuration, you must refer to the FAQ to adjust the learning rate and the number of iterations.

### 1.2 Dataset

The feature extraction model expands and optimizes the original training data, and finally uses the following 17 public datasets:

| Dataset      | Data Amount | Number of classes | Scenario |                               Dataset Address                                |
| :----------- | :---------: | :---------------: | :------: | :--------------------------------------------------------------------------: |
| Aliproduct   |   2498771   |       50030       |  goods   |  [link](https://retailvisionworkshop.github.io/recognition_challenge_2020/)  |
| GLDv2        |   1580470   |       81313       | landmark |           [link](https://github.com/cvdfoundation/google-landmark)           |
| VeRI-Wild    |   277797    |       30671       | vehicle  |                [link](https://github.com/PKU-IMRE/VERI-Wild)                 |
| LogoDet-3K   |   155427    |       3000        |   logo   |          [link](https://github.com/Wangjing1551/LogoDet-3K-Dataset)          |
| iCartoonFace |   389678    |       5013        | cartoon  | [link](http://challenge.ai.iqiyi.com/detail?raceId=5def69ace9fcf68aef76a75d) |
| SOP          |    59551    |       11318       |  goods   |          [link](https://cvgl.stanford.edu/projects/lifted_struct/)           |
| Inshop       |    25882    |       3997        |  goods   |        [link](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)         |
| **Total**    |   **6M**    |     **192K**      |    -     |                                      -                                       |


For the dataset of the subject detection model, please refer to [subject detection model dataset](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.5/docs/zh_CN/training/PP-ShiTu/mainbody_detection.md#1-%E6%95%B0%E6%8D%AE%E9%9B%86)

### Metrics

| Model Name                           | Model Introduction | Model Volume | Input Dimensions | ips |
| ------------------------------------ | ------------------ | ------------ | ---------------- | --- |
| picodet_lcnet_x2_5_640_mainbody.yml  | body detection     | 30MB         | 640              | 21  |
| GeneralRecognition_PPLCNet_x2_5.yaml | Feature extraction | 34MB         | 224              | 200 |


## 2. Inference Benchmark

### 2.1 Environment

* The inference speed test of the PP-ShiTu mainbody detection and feature extraction model uses CPU, with MKLDNN turned on, 10 threads, and batch size=1 for testing.

### 2.2 Dataset

The PP-ShiTu feature extraction model uses the self-built product dataset as the test set

### 2.3 Results

| model    | storage (mainbody detection + feature extraction) | product  |
| :------- | :----------------------------------------------- | :------- |
|          |                                                  | recall@1 |
| PP-ShiTu | 64(30+34)MB                                      | 66.8%    |


## 3. Related Instructions
Please refer to: https://github.com/PaddlePaddle/PaddleClas/tree/release/2.4/docs/zh_CN/image_recognition_pipeline
