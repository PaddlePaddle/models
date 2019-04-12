Fluid 模型库
============

语音识别
--------

自动语音识别（Automatic Speech Recognition, ASR）是将人类声音中的词汇内容转录成计算机可输入的文字的技术。语音识别的相关研究经历了漫长的探索过程，在HMM/GMM模型之后其发展一直较为缓慢，随着深度学习的兴起，其迎来了春天。在多种语言识别任务中，将深度神经网络(DNN)作为声学模型，取得了比GMM更好的性能，使得 ASR 成为深度学习应用最为成功的领域之一。而由于识别准确率的不断提高，有越来越多的语言技术产品得以落地，例如语言输入法、以智能音箱为代表的智能家居设备等 — 基于语言的交互方式正在深刻的改变人类的生活。

与 [DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech) 中深度学习模型端到端直接预测字词的分布不同，本实例更接近传统的语言识别流程，以音素为建模单元，关注语言识别中声学模型的训练，利用[kaldi](http://www.kaldi-asr.org) 进行音频数据的特征提取和标签对齐，并集成 kaldi 的解码器完成解码。

-  [DeepASR](https://github.com/PaddlePaddle/models/blob/develop/PaddleSpeech/DeepASR/README_cn.md)

