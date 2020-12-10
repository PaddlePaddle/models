# 样例代码运行指南

---
## 内容
参考本文档，您可以快速熟悉VideoTag的使用方法，观察VideoTag的预训练模型在示例视频上的预测结果。

文档内容包括:
- [安装说明](#安装说明)
- [数据准备](#数据准备)
- [模型推断](#模型推断)


## 安装说明

### 环境依赖：

```
    CUDA >= 9.0
    cudnn >= 7.5
```

### 依赖安装:

- 1.7.0 <= PaddlePaddle版本 <= 2.0.0: pip install paddlepaddle-gpu==1.8.4.post97 -i https://mirror.baidu.com/pypi/simple
- opencv版本 >= 4.1.0: pip install opencv-python==4.2.0.34

## 数据准备

### 预训练权重下载

我们提供了[TSN](https://videotag.bj.bcebos.com/video_tag_tsn.tar)和[AttentionLSTM](https://videotag.bj.bcebos.com/video_tag_lstm.tar)预训练权重，请在video\_tag目录下新建weights目录，并将下载解压后的参数文件放在weights目录下:

```
    mkdir weights
    cd weights
    wget https://videotag.bj.bcebos.com/video_tag_tsn.tar
    wget https://videotag.bj.bcebos.com/video_tag_lstm.tar
    tar -zxvf video_tag_tsn.tar
    tar -zxvf video_tag_lstm.tar
    rm video_tag_tsn.tar -rf
    rm video_tag_lstm.tar -rf
    mv video_tag_tsn/* .
    mv attention_lstm/* .
    rm video_tag_tsn/ -rf
    rm attention_lstm -rf
```

所得目录结构如下：

```
video_tag
  ├──weights
    ├── attention_lstm.pdmodel
    ├── attention_lstm.pdopt  
    ├── attention_lstm.pdparams
    ├── tsn.pdmodel
    ├── tsn.pdopt
    └── tsn.pdparams
```

### 示例视频下载

我们提供了[样例视频](https://videotag.bj.bcebos.com/mp4.tar)方便用户测试，请下载后解压，并将视频文件放置在video\_tag/data/mp4目录下:

```
    cd data/
    wget https://videotag.bj.bcebos.com/mp4.tar
    tar -zxvf mp4.tar
    rm mp4.tar -rf
```

所得目录结构如下：

```
video_tag
  ├──data
    ├── mp4
      ├── 1.mp4
      ├── 2.mp4
      └── ...
```

## 模型推断

模型推断的启动方式如下：

    python videotag_test.py

- 预测结果会以日志方式打印，示例如下:
```
[========video_id [ data/mp4/1.mp4 ] , topk(20) preds: ========]
class_id: 3110, class_name: 训练 ,  probability:  0.97730666399
class_id: 2159, class_name: 蹲 ,  probability:  0.945082366467
...
[========video_id [ data/mp4/2.mp4 ] , topk(20) preds: ========]
class_id: 2773, class_name: 舞蹈 ,  probability:  0.850423932076
class_id: 1128, class_name: 表演艺术 ,  probability:  0.0446354188025
...
```

- 通过--save\_dir可指定预测结果存储路径，默认为video\_tag/data/VideoTag\_results，不同输入视频的预测结果分文件保存在不同的json文件中，文件的内容格式为：

```
    [file_path,
     {"class_name": class_name1, "probability": probability1, "class_id": class_id1},
     {"class_name": class_name2, "probability": probability2, "class_id": class_id2},
     ...
    ]
```
