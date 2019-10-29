# BMN模型数据使用说明

BMN模型使用ActivityNet 1.3数据集，使用方法有如下两种方式：

方式一：

首先参考[下载说明](https://github.com/activitynet/ActivityNet/tree/master/Crawler)下载原始数据集。在训练此模型时，需要先使用TSN对源文件抽取特征。可以[自行抽取](https://github.com/yjxiong/temporal-segment-networks)视频帧及光流信息，预训练好的TSN模型可从[此处](https://github.com/yjxiong/anet2016-cuhk)下载。

方式二：

我们也在[百度网盘](https://pan.baidu.com/s/19GI3_-uZbd_XynUO6g-8YQ)和[谷歌云盘](https://drive.google.com/file/d/1ISemndlSDS2FtqQOKL0t3Cjj9yk2yznF/view?usp=sharing)提供了处理好的视频特征。若使用百度网盘下载，在解压前请使用如下命令：

    cat zip_csv_mean_100.z* > csv_mean_100.zip

解压完成后，请相应修改configs/bmn.yaml文件中的特征路径feat\_path。
