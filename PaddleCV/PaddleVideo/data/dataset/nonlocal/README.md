# Non-local模型数据说明

在Non-local模型中，输入数据是mp4文件，在reader部分的代码中，使用opencv读取mp4文件对视频进行解码和采样。train和valid数据随机选取起始帧的位置，对每帧图像做随机增强，短边缩放至[256, 320]之间的某个随机数，长边根据长宽比计算出来，截取出224x224大小的区域。test时每条视频会选取10个不同的位置作为起始帧，同时会选取三个不同的空间位置作为crop区域的起始点，这样每个视频会进行10x3次采样，对这30个样本的预测概率求和，选取概率最大的分类作为最终的预测结果。

## 数据下载

下载kinetics400数据，具体方法见[数据说明](../README.md)中kinetics数据部分，假设下载的mp4文件存放在DATADIR目录下，train和validation数据分别位于$DATADIR/train和$DATADIR/valid目录。在下载数据的时候，将所有视频的高度缩放至256，宽度通过长宽比计算出来。

## 下载官方数据列表

将官方提供的数据集文件表格[kinetics-400\_train.csv](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics/data/kinetics-400_train.csv)和[kinetics-400\_val.csv](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics/data/kinetics-400_val.csv)下载到此目录。

## 生成文件列表

运行下面的代码即可生成trainlist.txt、vallist.txt和testlist.txt，

    python generate_filelist.py ${TRAIN_DIR} ${VALID_DIR}

其中TRAIN\_DIR和VALID\_DIR分别是存放训练和验证数据集文件的路径。注意请确认[kinetics-400\_train.csv](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics/data/kinetics-400_train.csv)已经下载到本地，不然运行generate\_filelist.py时会报错。

另外，如果要观察模型推断的效果，可以复制testlist.txt生成inferlist.txt，

    cp testlist.txt inferlist.txt

生成inferlist.txt。也可以在predict的时候指定`video_path`对单个视频文件进行预测。
