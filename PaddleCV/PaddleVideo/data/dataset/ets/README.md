# ETS模型数据使用说明

ETS模型使用ActivityNet Captions数据集，数据准备方法如下：

步骤一. 特征数据准备:

- 在[ActivityNet下载页面](http://activity-net.org/challenges/2019/tasks/anet_captioning.html)中，下载"Frame-level features"特征数据集(~89GB)。将下载好的resnet152i\_features\_activitynet\_5fps\_320x240.pkl数据文件存放在PaddleVideo/data/dataset/ets目录下；

- 运行PaddleVideo/data/dataset/ets/generate\_train\_pickle.py文件，将数据转化为pickle文件，便于内存载入。生成的数据存放在PaddleVideo/data/dataset/ets/feat\_data文件夹下。

步骤二. 标签及索引数据准备：

- 在[Dense-Captioning Events in Videos项目页面](http://cs.stanford.edu/people/ranjaykrishna/densevid/)，从dataset链接中下载captions文件夹，其中包含标签和索引的json文件。将captions文件夹存放在PaddleVideo/data/dataset/ets目录下；

- 按[数据评估](../../../metrics/ets\_metrics/README.md)步骤下载好coco-caption文件夹，并将其放置在PaddleVideo目录下；

- python运行generate\_data.py文件，生成训练用的文本文件train.list和val.list。

步骤三. 生成infer数据：

- 完成前两个步骤后，python运行generate\_infer\_data.py文件可生成infer.list文件。

按如上步骤操作，最终PaddleVideo/data/dataset/ets的目录结构为：

```
ets
  |
  |----feat_data/
  |----train.list
  |----val.list
  |----generate_train_pickle.py
  |----generate_data.py
  |----generate_infer_data.py
  |----captions/
  |----resnet152_features_activitynet_5fps_320x240.pkl (生成feat_data后可移除以节省磁盘空间)
```
