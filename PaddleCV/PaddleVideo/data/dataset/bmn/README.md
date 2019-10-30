# BMN模型数据使用说明

BMN模型使用ActivityNet 1.3数据集，使用方法有如下两种方式：

方式一：

首先参考[下载说明](https://github.com/activitynet/ActivityNet/tree/master/Crawler)下载原始数据集。在训练此模型时，需要先使用TSN对源文件抽取特征。可以[自行抽取](https://github.com/yjxiong/temporal-segment-networks)视频帧及光流信息，预训练好的TSN模型可从[此处](https://github.com/yjxiong/anet2016-cuhk)下载。

方式二：

我们也提供了处理好的视频特征，请下载[bmn\_feat](https://paddlemodels.bj.bcebos.com/video_detection/bmn_feat.tar.gz)数据后解压，同时相应的修改configs/bmn.yaml文件中的特征路径feat\_path。
