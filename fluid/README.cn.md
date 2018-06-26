## 语音识别


语音识别（speech recognition）是将人类声音中的词汇内容转录成计算机可输入的文字的技术。语音识别的相关研究经历了漫长的探索过程，在HMM/GMM模型之后其发展一直较为缓慢，由于深度学习的兴起，语音识别迎来了春天，成为深度学习应用最为成功的领域之一。随着识别准确率的不断提高，有越来越多的语言技术产品得以落地，例如语言输入法、以智能音箱为代表的智能家居设备等 —— 基于语言的交互方式正在深刻的改变人类的生活。

与 [DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech) 中深度学习模型端到端直接预测字词的分布不同，本实例关注语言识别中声学模型的训练，以音素为建模单元，利用kaldi进行音频数据的特征提取，并集成kaldi的解码器完成解码。


- [DeepASR](https://github.com/PaddlePaddle/models/tree/develop/fluid/DeepASR)

## 机器翻译

机器翻译（machine translation）将一种自然语言(源语言)转换成一种自然语言（目标语音），是自然语言处理中非常基础和重要的研究方向。在全球化的浪潮中，机器翻译在促进跨语言文明的交流中所起的重要作用是不言而喻的。其发展经历了统计机器翻译，和基于神经网络的神经机器翻译(Nueural Machine Translation, NMT)等阶段。在 NMT 成熟后，机器翻译才真正得以大规模应用。而早阶段的 NMT 主要是基于循环神经网络 RNN 的，其训练过程后时间步依赖于前时间步的计算，时间步之间难以并行化以提高训练速度。因此，非 RNN 结构的 NMT 得以应运而生，例如基于卷积神经网络 CNN 的结构和基于注意力机制（Attention）的结构。

本实例所实现的Transformer就是一个基于全注意力机制的机器翻译模型，其中不再有RNN或者CNN等模型结构，而是利用 Attention 学习源语言中的上下文依赖，最终在多个数据集上取得了最好的翻译效果。

- [Transformer](https://github.com/PaddlePaddle/models/tree/develop/fluid/neural_machine_translation/transformer)

## 强化学习

强化学习是近年来一个愈发重要的机器学习方向，特别是与深度学习相结合而形成的深度强化学习，取得了很多令人惊异的成就，人们所熟知的战胜顶级围棋职业选手的 AlphaGo 就是一个典型的例子。

当前深度强化学习最成功的应用是游戏领域，本实例利用Fluid这个灵活的框架，实现了在Atari游戏中表现优异的几个重要工作。

- [DeepQNetwork](https://github.com/PaddlePaddle/models/tree/develop/fluid/DeepQNetwork)
