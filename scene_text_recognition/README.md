运行本目录下的程序示例需要使用PaddlePaddle v0.10.0 版本。如果您的PaddlePaddle安装版本低于此要求，请按照[安装文档](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html)中的说明更新PaddlePaddle安装版本。

---

# 场景文字识别 (STR, Scene Text Recognition)

## STR任务简介

许多场景图像中包含着丰富的文本信息，它们可以从很大程度上帮助人们去认知场景图像的内容及含义，因此场景图像中的文本识别对所在图像的信息获取具有极其重要的作用。同时，场景图像文字识别技术的发展也促进了一些新型应用的产生，例如：\[[1](#参考文献)\]通过使用深度学习模型来自动识别路牌中的文字，帮助街景应用获取更加准确的地址信息。

本例将演示如何用 PaddlePaddle 完成 **场景文字识别 (STR, Scene Text Recognition)** 。任务如下图所示，给定一张场景图片，`STR` 需要从中识别出对应的文字"keep"。

<p align="center">
<img src="./images/503.jpg"/><br/>
图 1. 输入数据示例 "keep"
</p>


## 使用 PaddlePaddle 训练与预测

### 安装依赖包
```bash
pip install -r requirements.txt
```

### 修改配置参数

 `config.py` 脚本中包含了模型配置和训练相关的参数以及对应的详细解释，代码片段如下：
```python
class TrainerConfig(object):

      # Whether to use GPU in training or not.
      use_gpu = True
      # The number of computing threads.
      trainer_count = 1

      # The training batch size.
      batch_size = 10

      ...


class ModelConfig(object):

      # Number of the filters for convolution group.
      filter_num = 8

      ...
```

修改 `config.py` 脚本可以实现对参数的调整。例如，通过修改 `use_gpu` 参数来指定是否使用 GPU 进行训练。

### 模型训练
训练脚本 [./train.py](./train.py) 中设置了如下命令行参数：

```
Options:
  --train_file_list_path TEXT  The path of the file which contains path list
                               of train image files.  [required]
  --test_file_list_path TEXT   The path of the file which contains path list
                               of test image files.  [required]
  --label_dict_path TEXT       The path of label dictionary. If this parameter
                               is set, but the file does not exist, label
                               dictionay will be built from the training data
                               automatically.  [required]
  --model_save_dir TEXT        The path to save the trained models (default:
                               'models').
  --help                       Show this message and exit.

```

- `train_file_list` ：训练数据的列表文件，每行由图片的存储路径和对应的标记文本组成，格式为：
```
word_1.png, "PROPER"
word_2.png, "FOOD"
```
- `test_file_list` ：测试数据的列表文件，格式同上。
- `label_dict_path` ：训练数据中标记字典的存储路径，如果指定路径中字典文件不存在，程序会使用训练数据中的标记数据自动生成标记字典。
- `model_save_dir` ：模型参数的保存目录，默认为`./models`。

### 具体执行的过程：

1.从官方网站下载数据\[[2](#参考文献)\]（Task 2.3: Word Recognition (2013 edition)），会有三个文件: `Challenge2_Training_Task3_Images_GT.zip`、`Challenge2_Test_Task3_Images.zip` 和 `Challenge2_Test_Task3_GT.txt`。
分别对应训练集的图片和图片对应的单词、测试集的图片、测试数据对应的单词。然后执行以下命令，对数据解压并移动至目标文件夹：

```bash
mkdir -p data/train_data
mkdir -p data/test_data
unzip Challenge2_Training_Task3_Images_GT.zip -d data/train_data
unzip Challenge2_Test_Task3_Images.zip -d data/test_data
mv Challenge2_Test_Task3_GT.txt data/test_data
```

2.获取训练数据文件夹中 `gt.txt` 的路径 (data/train_data）和测试数据文件夹中`Challenge2_Test_Task3_GT.txt`的路径(data/test_data)。

3.执行如下命令进行训练：
```bash
python train.py \
--train_file_list_path 'data/train_data/gt.txt' \
--test_file_list_path 'data/test_data/Challenge2_Test_Task3_GT.txt' \
--label_dict_path 'label_dict.txt'
```
4.训练过程中，模型参数会自动备份到指定目录，默认会保存在 `./models` 目录下。


### 预测
预测部分由 `infer.py` 完成，使用的是最优路径解码算法，即：在每个时间步选择一个概率最大的字符。在使用过程中，需要在 `infer.py` 中指定具体的模型保存路径、图片固定尺寸、batch_size（默认为10）、标记词典路径和图片文件的列表文件。执行如下代码：
```bash
python infer.py \
--model_path 'models/params_pass_00000.tar.gz' \
--image_shape '173,46' \
--label_dict_path 'label_dict.txt' \
--infer_file_list_path 'data/test_data/Challenge2_Test_Task3_GT.txt'
```
即可进行预测。

### 其他数据集

-   [SynthText in the Wild Dataset](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)(41G)
-   [ICDAR 2003 Robust Reading Competitions](http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2003_Robust_Reading_Competitions)

### 注意事项

- 由于模型依赖的 `warp CTC` 只有CUDA的实现，本模型只支持 GPU 运行。
- 本模型参数较多，占用显存比较大，实际执行时可以通过调节 `batch_size` 来控制显存占用。
- 本例使用的数据集较小，如有需要，可以选用其他更大的数据集\[[3](#参考文献)\]来训练模型。

## 参考文献

1. [Google Now Using ReCAPTCHA To Decode Street View Addresses](https://techcrunch.com/2012/03/29/google-now-using-recaptcha-to-decode-street-view-addresses/)
2. [Focused Scene Text](http://rrc.cvc.uab.es/?ch=2&com=introduction)
3. [SynthText in the Wild Dataset](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
