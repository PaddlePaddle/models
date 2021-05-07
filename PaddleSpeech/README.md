PaddleSpeech 语音模型库
============

语音识别
--------

自动语音识别（Automatic Speech Recognition, ASR）是将人类声音中的词汇内容转录成计算机可输入的文字的技术。语音识别的相关研究经历了漫长的探索过程，在 HMM/GMM 模型之后其发展一直较为缓慢，随着深度学习的兴起，其迎来了春天。在多种语言识别任务中，将深度神经网络 (DNN) 作为声学模型，取得了比 GMM 更好的性能，使得 ASR 成为深度学习应用非常成功的领域之一。而由于识别准确率的不断提高，有越来越多的语言技术产品得以落地，例如语言输入法、以智能音箱为代表的智能家居设备等 — 基于语言的交互方式正在深刻的改变人类的生活。

-  [DeepASR](https://github.com/PaddlePaddle/models/blob/develop/PaddleSpeech/DeepASR/README_cn.md) 本实例更接近传统的语言识别流程，以音素为建模单元，关注语言识别中声学模型的训练，利用 [kaldi](http://www.kaldi-asr.org) 进行音频数据的特征提取和标签对齐，并集成 kaldi 的解码器完成解码。

- [DeepSpeech2](https://github.com/PaddlePaddle/DeepSpeech) 是一个采用 PaddlePaddle 平台的端到端自动语音识别（ASR）引擎的开源项目，具体原理请参考这篇论文 [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595)。

## 语音合成

语音合成 (Speech Synthesis) 技术是指用人工方法合成可辨识的语音。文本转语音 (Text-To-Speech) 系统是对语音合成技术的具体应用，其任务是给定某种语言的文本，合成对应的语音。语音合成技术是基于语音的人机交互，实时语音翻译等技术的基础。传统的文本转语音模型分为文本到音位，音位到频谱，频谱到波形等几个阶段分别进行优化，而随着深度学习技术在语音技术的应用的发展，端到端的文本转语音模型正在取得快速发展。

- [Parakeet](https://github.com/PaddlePaddle/Parakeet) (Paddle PARAllel text-to-speech toolKIT) 是一个定位于灵活、高效的语音合成工具集，支持多个前沿的语音合成模型，包括 WaveFlow、ClariNet、WaveNet、Deep Voice 3、Transformer TTS、FastSpeech 等。
