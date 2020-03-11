# C-TCN模型数据使用说明

C-TCN模型使用ActivityNet 1.3数据集，具体下载方法请参考官方[下载说明](http://activity-net.org/index.html)。在训练此模型时，需要先对mp4源文件抽取RGB和Flow特征，然后再用训练好的TSN模型提取出抽象的特征数据，并存储为pickle文件格式。我们使用百度云提供转化后的数据[下载链接](https://paddlemodels.bj.bcebos.com/video_detection/CTCN_data.tar.gz)。转化后的数据文件目录结构为：

```
data
  |
  |----senet152-201cls-flow-60.9-5seg-331data\_train
  |----senet152-201cls-rgb-70.3-5seg-331data\_331img\_train
  |----senet152-201cls-flow-60.9-5seg-331data\_val
  |----senet152-201cls-rgb-70.3-5seg-331data\_331img\_val
```

同时需要下载如下几个数据文件Activity1.3\_train\_rgb.listformat, Activity1.3\_val\_rgb.listformat, labels.txt, val\_duration\_frame.list，并放到dataset/ctcn目录下。
