## 1. Train Benchmark

### 1.1 Software and hardware environment

* The feature extraction model of PP-ShiTuV2 uses 8 GPUs in the training process, the batch size of each GPU is 256 for training, the sampler uses PKSampler, a mini-batch with 256 samples has 64 classes, and each class contains 4 different pictures. If the number of training GPUs and batch size are not consistent with the above configuration, you must refer to the FAQ to adjust the learning rate and the number of iterations.

  **Note**: Since this model uses PKSampler and metric learning methods, changing the batch size may have a significant impact on performance.

* 8 GPUs are used in the training process of the detection model of PP-ShiTuV2, and the batch size of each GPU is 28 for training. If the number of training GPUs and batch size are not consistent with the above configuration, you must refer to the FAQ to adjust the learning rate and the number of iterations.

### 1.2 Dataset

The feature extraction model expands and optimizes the original training data, and finally uses the following 17 public datasets:

| Dataset                | Data Amount | Number of classes |  Scenario   |                                     Dataset Address                                     |
| :--------------------- | :---------: | :---------------: | :---------: | :-------------------------------------------------------------------------------------: |
| Aliproduct             |   2498771   |       50030       | Commodities |      [Address](https://retailvisionworkshop.github.io/recognition_challenge_2020/)      |
| GLDv2                  |   1580470   |       81313       |  Landmark   |               [address](https://github.com/cvdfoundation/google-landmark)               |
| VeRI-Wild              |   277797    |       30671       |  Vehicles   |                    [Address](https://github.com/PKU-IMRE/VERI-Wild)                     |
| LogoDet-3K             |   155427    |       3000        |    Logo     |              [Address](https://github.com/Wangjing1551/LogoDet-3K-Dataset)              |
| SOP                    |    59551    |       11318       | Commodities |              [Address](https://cvgl.stanford.edu/projects/lifted_struct/)               |
| Inshop                 |    25882    |       3997        | Commodities |            [Address](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)             |
| bird400                |    58388    |        400        |    birds    |          [address](https://www.kaggle.com/datasets/gpiosenka/100-bird-species)          |
| 104flows               |    12753    |        104        |   Flowers   |              [Address](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)              |
| Cars                   |    58315    |        112        |  Vehicles   |            [Address](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)            |
| Fashion Product Images |    44441    |        47         |  Products   | [Address](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) |
| flowerrecognition      |    24123    |        59         |   flower    |         [address](https://www.kaggle.com/datasets/aymenktari/flowerrecognition)         |
| food-101               |   101000    |        101        |    food     |         [address](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)          |
| fruits-262             |   225639    |        262        |   fruits    |            [address](https://www.kaggle.com/datasets/aelchimminut/fruits262)            |
| inaturalist            |   265213    |       1010        |   natural   |           [address](https://github.com/visipedia/inat_comp/tree/master/2017)            |
| indoor-scenes          |    15588    |        67         |   indoor    |       [address](https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019)       |
| Products-10k           |   141931    |       9691        |  Products   |                       [Address](https://products-10k.github.io/)                        |
| CompCars               |    16016    |        431        |  Vehicles   |     [Address](http://​​​​​​http://ai.stanford.edu/~jkrause/cars/car_dataset.html​)      |
| **Total**              |   **6M**    |     **192K**      |      -      |                                            -                                            |


For the dataset of the mainbody detection model, please refer to [mainbody detection model dataset](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.5/docs/zh_CN/training/PP-ShiTu/mainbody_detection.md#1-%E6%95%B0%E6%8D%AE%E9%9B%86)

### Metrics

| Model Name                               | Model Introduction | Model Volume           | Input Dimensions | ips |
| ---------------------------------------- | ------------------ | ---------------------- | ---------------- | --- |
| picodet_lcnet_x2_5_640_mainbody.yml      | body detection     | 30MB                   | 640              | 21  |
| GeneralRecognitionV2_PPLCNetV2_base.yaml | Feature extraction | 19MB (KL quantization) | 224              | 163 |


## 2. Inference Benchmark

### 2.1 Environment

* The inference speed test of the PP-ShiTuV2 mainbody detection and feature extraction model uses CPU, with MKLDNN turned on, 10 threads, and batch size=1 for testing.

### 2.2 Dataset

The PP-ShiTuV2 feature extraction model uses the self-built product dataset as the test set

### 2.3 Results

| model      | storage (mainbody detection + feature extraction) | product  |
| :--------- | :----------------------------------------------- | :------- |
|            |                                                  | recall@1 |
| PP-ShiTuV1 | 64(30+34)MB                                      | 66.8%    |
| PP-ShiTuV2 | 49(30+19)MB                                      | 73.8%    |


## 3. Related Instructions
Please refer to: https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PP-ShiTu/README.md
