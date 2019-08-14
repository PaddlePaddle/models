# 数据使用说明

- [Youtube-8M](#Youtube-8M数据集)
- [Kinetics](#Kinetics数据集)
- [Non-local](#Non-local)
- [C-TCN](#C-TCN)

## Youtube-8M数据集
这里用到的是YouTube-8M 2018年更新之后的数据集。使用官方数据集，并将TFRecord文件转化为pickle文件以便PaddlePaddle使用。Youtube-8M数据集官方提供了frame-level和video-level的特征，这里只需使用到frame-level的特征。

### 数据下载
请使用Youtube-8M官方链接分别下载[训练集](http://us.data.yt8m.org/2/frame/train/index.html)和[验证集](http://us.data.yt8m.org/2/frame/validate/index.html)。每个链接里各提供了3844个文件的下载地址，用户也可以使用官方提供的[下载脚本](https://research.google.com/youtube8m/download.html)下载数据。数据下载完成后，将会得到3844个训练数据文件和3844个验证数据文件（TFRecord格式）。
假设存放视频模型代码库PaddleVideo的主目录为: Code\_Root，进入data/dataset/youtube8m目录

    cd data/dataset/youtube8m

在youtube8m下新建目录tf/train和tf/val

    mkdir tf && cd tf

    mkdir train && mkdir val

并分别将下载的train和validate数据存放在其中。

### 数据格式转化

为了适用于PaddlePaddle训练，需要离线将下载好的TFRecord文件格式转成了pickle格式，转换脚本请使用[dataset/youtube8m/tf2pkl.py](./youtube8m/tf2pkl.py)。

在data/dataset/youtube8m 目录下新建目录pkl/train和pkl/val

    cd data/dataset/youtube8m

    mkdir pkl && cd pkl

    mkdir train && mkdir val


转化文件格式(TFRecord -> pkl)，进入data/dataset/youtube8m目录，运行脚本

    python tf2pkl.py ./tf/train ./pkl/train

和

    python tf2pkl.py ./tf/val ./pkl/val

分别将train和validate数据集转化为pkl文件。tf2pkl.py文件运行时需要两个参数，分别是数据源tf文件存放路径和转化后的pkl文件存放路径。

备注：由于TFRecord文件的读取需要用到Tensorflow，用户要先安装Tensorflow，或者在安装有Tensorflow的环境中转化完数据，再拷贝到data/dataset/youtube8m/pkl目录下。为了避免和PaddlePaddle环境冲突，建议先在其他地方转化完成再将数据拷贝过来。

### 生成文件列表

进入data/dataset/youtube8m目录

    cd $Code_Root/data/dataset/youtube8m

    ls $Code_Root/data/dataset/youtube8m/pkl/train/* > train.list

    ls $Code_Root/data/dataset/youtube8m/pkl/val/* > val.list

    ls $Code_Root/data/dataset/youtube8m/pkl/val/* > test.list

    ls $Code_Root/data/dataset/youtube8m/pkl/val/* > infer.list

此处Code\_Root是PaddleVideo目录所在的绝对路径，比如/ssd1/user/models/PaddleCV/PaddleVideo，在data/dataset/youtube8m目录下将生成四个文件，train.list，val.list，test.list和infer.list，每一行分别保存了一个pkl文件的绝对路径，示例如下：

    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/youtube8m/pkl/train/train0471.pkl
    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/youtube8m/pkl/train/train0472.pkl
    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/youtube8m/pkl/train/train0473.pkl
    ...

或者

    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/youtube8m/pkl/val/validate3666.pkl
    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/youtube8m/pkl/val/validate3666.pkl
    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/youtube8m/pkl/val/validate3666.pkl
    ...

- 备注：由于Youtube-8M数据集中test部分的数据没有标签，所以此处使用validate数据做模型评估。

## Kinetics数据集

Kinetics数据集是DeepMind公开的大规模视频动作识别数据集，有Kinetics400与Kinetics600两个版本。这里使用Kinetics400数据集，具体的数据预处理过程如下。

### mp4视频下载
在Code\_Root目录下创建文件夹

    cd $Code_Root/data/dataset && mkdir kinetics

    cd kinetics && mkdir data_k400 && cd data_k400

    mkdir train_mp4 && mkdir val_mp4

ActivityNet官方提供了Kinetics的下载工具，具体参考其[官方repo ](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics)即可下载Kinetics400的mp4视频集合。将kinetics400的训练与验证集合分别下载到data/dataset/kinetics/data\_k400/train\_mp4与data/dataset/kinetics/data\_k400/val\_mp4。

### mp4文件预处理

为提高数据读取速度，提前将mp4文件解帧并打pickle包，dataloader从视频的pkl文件中读取数据（该方法耗费更多存储空间）。pkl文件里打包的内容为(video-id,[frame1, frame2,...,frameN],label)。

在 data/dataset/kinetics/data\_k400目录下创建目录train\_pkl和val\_pkl

    cd $Code_Root/data/dataset/kinetics/data_k400

    mkdir train_pkl && mkdir val_pkl

进入$Code\_Root/data/dataset/kinetics目录，使用video2pkl.py脚本进行数据转化。首先需要下载[train](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics/data/kinetics-400_train.csv)和[validation](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics/data/kinetics-400_val.csv)数据集的文件列表。

首先生成预处理需要的数据集标签文件

    python generate_label.py kinetics-400_train.csv kinetics400_label.txt

然后执行如下程序：

    python video2pkl.py kinetics-400_train.csv $Source_dir $Target_dir  8 #以8个进程为例

- 该脚本依赖`ffmpeg`库，请预先安装`ffmpeg`

对于train数据，

    Source_dir = $Code_Root/data/dataset/kinetics/data_k400/train_mp4

    Target_dir = $Code_Root/data/dataset/kinetics/data_k400/train_pkl

对于val数据，

    Source_dir = $Code_Root/data/dataset/kinetics/data_k400/val_mp4

    Target_dir = $Code_Root/data/dataset/kinetics/data_k400/val_pkl

这样即可将mp4文件解码并保存为pkl文件。

### 生成训练和验证集list

    cd $Code_Root/data/dataset/kinetics

    ls $Code_Root/data/dataset/kinetics/data_k400/train_pkl/* > train.list

    ls $Code_Root/data/dataset/kinetics/data_k400/val_pkl/* > val.list

    ls $Code_Root/data/dataset/kinetics/data_k400/val_pkl/* > test.list

    ls $Code_Root/data/dataset/kinetics/data_k400/val_pkl/* > infer.list

即可生成相应的文件列表，train.list和val.list的每一行表示一个pkl文件的绝对路径，示例如下：

    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/kinetics/data_k400/train_pkl/data_batch_100-097
    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/kinetics/data_k400/train_pkl/data_batch_100-114
    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/kinetics/data_k400/train_pkl/data_batch_100-118
    ...

或者

    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/kinetics/data_k400/val_pkl/data_batch_102-085
    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/kinetics/data_k400/val_pkl/data_batch_102-086
    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/kinetics/data_k400/val_pkl/data_batch_102-090
    ...


## Non-local

Non-local模型也使用kinetics数据集，不过其数据处理方式和其他模型不一样，详细内容见[Non-local数据说明](./nonlocal/README.md)

## C-TCN

C-TCN模型使用ActivityNet 1.3数据集，具体使用方法见[C-TCN数据说明](./ctcn/README.md)
