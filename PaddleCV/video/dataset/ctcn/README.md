# C-TCN模型数据使用说明

C-TCN模型使用ActivityNet 1.3数据集，具体下载方法请参考官方下载说明。下载之后的数据集结构如下：

data
  |
  |----senet152-201cls-flow-60.9-5seg-331data\_train
  |----senet152-201cls-rgb-70.3-5seg-331data\_331img\_train
  |----senet152-201cls-flow-60.9-5seg-331data\_val
  |----senet152-201cls-rgb-70.3-5seg-331data\_331img\_val

同时需要下载如下几个数据文件Activity1.3\_train\_rgb.listformat, Activity1.3\_val\_rgb.listformat, labels.txt, test\_val\_label.list, val\_duration\_frame.list，并放到dataset/ctcn目录下
