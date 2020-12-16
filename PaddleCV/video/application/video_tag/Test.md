# 预训练模型自测指南

## 内容
参考本文档，您可以快速测试VideoTag的预训练模型在自己业务数据上的预测效果。

主要内容包括:
- [数据准备](#数据准备)
- [模型推断](#模型推断)

## 数据准备

在数据准备阶段，您需要准备好自己的测试数据，并在video\_tag/data/VideoTag\_test.list文件中指定待推断的测试文件路径，内容格式如下:
```
my_video_path/my_video_file1.mp4
my_video_path/my_video_file2.mp4
...
```

## 模型推断

模型推断的启动方式如下：

    python videotag_test.py

- 目前支持的视频文件输入格式为：mp4、mkv和webm格式；

- 模型会从输入的视频文件中*均匀抽取300帧*用于预测。对于较长的视频文件，建议先截取有效部分输入模型以提高预测速度；

- 通过--use\_gpu参数可指定是否使用gpu进行推断，默认使用gpu。对于10s左右的短视频文件，gpu推断时间约为4s；

- 通过--filelist可指定输入list文件路径，默认为video\_tag/data/VideoTag\_test.list。
