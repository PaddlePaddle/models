# TALL模型数据使用说明

TALL模型使用TACoS数据集，数据准备过程如下：

步骤一. 训练和测试集：

- 训练和测试使用提取好的数据特征，请参考TALL模型原作者提供的[数据下载](https://github.com/jiyanggao/TALL)方法进行模型训练与评估；

步骤二. infer数据

- 为便于用户使用模型进行推断，我们提供了生成infer数据的文件./gen\_infer.py，执行完步骤一后python运行该文件便可在当前文件夹下生成infer数据。

按如上步骤操作，最终PaddleVideo/data/dataset/tall需要包含的文件有：

```
tall
  |
  |----Interval64_128_256_512_overlap0.8_c3d_fc6/
  |----Interval128_256_overlap0.8_c3d_fc6/
  |----train_clip-sentvec.pkl
  |----test_clip-sentvec.pkl
  |----video_allframes_info.pkl
  |----infer
         |
         |----infer_feat/
         |----infer_clip-sen.pkl
```
