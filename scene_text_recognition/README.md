# 场景文字识别 (STR, Scene Text Recognition)

## STR任务简介

在现实生活中，包括路牌、菜单、大厦标语在内的很多场景均会有文字出现，这些场景的照片中的文字为图片场景的理解提供了更多信息，\[[1](#参考文献)\]使用深度学习模型自动识别路牌中的文字，帮助街景应用获取更加准确的地址信息。

本文将针对 **场景文字识别 (STR, Scene Text Recognition)** 任务，演示如何用 PaddlePaddle 实现 一个端对端 CTC 的模型 **CRNN(Convolutional Recurrent Neural Network)**
\[[2](#参考文献)\]，具体的，本文使用如下图片进行训练，需要识别文字对应的文字 "keep"。

<p align="center">
<img src="./images/503.jpg"/><br/>
图 1. 数据示例 "keep"
</p>

## 使用 PaddlePaddle 训练与预测

### 模型训练
训练脚本参照 [./train.py](./train.py)，设置了如下命令行参数：

```
usage: train.py [-h] --image_shape IMAGE_SHAPE --train_file_list
                TRAIN_FILE_LIST --test_file_list TEST_FILE_LIST
                [--batch_size BATCH_SIZE]
                [--model_output_prefix MODEL_OUTPUT_PREFIX]
                [--trainer_count TRAINER_COUNT]
                [--save_period_by_batch SAVE_PERIOD_BY_BATCH]
                [--num_passes NUM_PASSES]

PaddlePaddle CTC example

optional arguments:
  -h, --help            show this help message and exit
  --image_shape IMAGE_SHAPE
                        image's shape, format is like '173,46'
  --train_file_list TRAIN_FILE_LIST
                        path of the file which contains path list of train
                        image files
  --test_file_list TEST_FILE_LIST
                        path of the file which contains path list of test
                        image files
  --batch_size BATCH_SIZE
                        size of a mini-batch
  --model_output_prefix MODEL_OUTPUT_PREFIX
                        prefix of path for model to store (default:
                        ./model.ctc)
  --trainer_count TRAINER_COUNT
                        number of training threads
  --save_period_by_batch SAVE_PERIOD_BY_BATCH
                        save model to disk every N batches
  --num_passes NUM_PASSES
                        number of passes to train (default: 1)
```

其中最重要的几个参数包括：

- `image_shape` 图片的尺寸
- `train_file_list` 训练数据的列表文件，每行一个路径加对应的text，格式类似：
```
word_1.png, "PROPER"
```
- `test_file_list` 测试数据的列表文件，格式同上

### 预测
预测部分由infer.py完成，本示例对于ctc的预测使用的是最优路径解码算法(CTC greedy decoder)，即在每个时间步选择一个概率最大的字符。在使用过程中，需要在infer.py中指定具体的模型目录、图片固定尺寸、batch_size和图片文件的列表文件。例如：
```python
model_path = "model.ctc-pass-9-batch-150-test-10.0065517931.tar.gz"  
image_shape = "173,46"
batch_size = 50
infer_file_list = 'data/test_data/Challenge2_Test_Task3_GT.txt'
```
然后运行```python infer.py```


### 具体执行的过程：

1.从官方下载数据\[[3](#参考文献)\]（Task 2.3: Word Recognition (2013 edition)），会有三个文件: Challenge2_Training_Task3_Images_GT.zip、Challenge2_Test_Task3_Images.zip和 Challenge2_Test_Task3_GT.txt。
分别对应训练集的图片和图片对应的单词，测试集的图片，测试数据对应的单词，然后执行以下命令，对数据解压并移动至目标文件夹：

```
mkdir -p data/train_data
mkdir -p data/test_data
unzip Challenge2_Training_Task3_Images_GT.zip -d data/train_data
unzip Challenge2_Test_Task3_Images.zip -d data/test_data
mv Challenge2_Test_Task3_GT.txt data/test_data
```

2.获取训练数据文件夹中 `gt.txt` 的路径 (data/train_data）和测试数据文件夹中`Challenge2_Test_Task3_GT.txt`的路径(data/test_data)

3.执行命令
```
python train.py --train_file_list data/train_data/gt.txt --test_file_list data/test_data/Challenge2_Test_Task3_GT.txt --image_shape '173,46'
```
4.训练过程中，模型参数会自动备份到指定目录，默认为 ./model.ctc

5.设置infer.py中的相关参数(模型所在路径)，运行```python infer.py``` 进行预测


### 其他数据集

-   [SynthText in the Wild Dataset](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)(41G)
-   [ICDAR 2003 Robust Reading Competitions](http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2003_Robust_Reading_Competitions)

### 注意事项

- 由于模型依赖的 `warp CTC` 只有CUDA的实现，本模型只支持 GPU 运行
- 本模型参数较多，占用显存比较大，实际执行时可以调节batch_size 控制显存占用
- 本模型使用的数据集较小，可以选用其他更大的数据集\[[4](#参考文献)\]来训练需要的模型

## 参考文献

1. [Google Now Using ReCAPTCHA To Decode Street View Addresses](https://techcrunch.com/2012/03/29/google-now-using-recaptcha-to-decode-street-view-addresses/)
2. Shi B, Bai X, Yao C. [An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition](https://arxiv.org/pdf/1507.05717.pdf)[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2016. APA
3. [Focused Scene Text](http://rrc.cvc.uab.es/?ch=2&com=introduction)
4. [SynthText in the Wild Dataset](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
