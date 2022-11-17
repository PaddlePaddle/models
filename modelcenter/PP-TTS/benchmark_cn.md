## 1. 训练 Benchmark

### 1.1 软硬件环境

* FastSpeech2 模型训练过程中使用 2 GPUs，每 GPU batch size为 64 进行训练。
* HiFiGAN 模型训练过程中使用 1 GPU，每 GPU batch size为 16 进行训练。
* python 版本: 3.7.0
* paddle 版本: v2.4.0rc0
* 机器: 8x Tesla V100-SXM2-32GB, 24 core Intel(R) Xeon(R) Gold 6148, 100Gbps RDMA network


### 1.2 数据集

| 语言 | 数据集 |音频信息 | 描述 |
| -------- | -------- | -------- | -------- |
| 中文 | [CSMSC](https://www.data-baker.com/open_source.html) | 48KHz, 16bit | 单说话人，女声，约12小时，具有高音频质量 |
| 中文 | [AISHELL-3](http://www.aishelltech.com/aishell_3) | 44.1kHz，16bit | 多说话人（218人），约85小时，音频质量不一致（有的说话人音频质量较高）|
| 英文 | [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/) | 22050Hz, 16bit | 单说话人，女声，约24小时，具有高音频质量|
| 英文 | [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) | 48kHz, 16bit | 多说话人（110人），约44小时，音频质量不一致（有的说话人音频质量较高）|

### 1.3 指标

|模型名称 | 模型简介 | 模型体积 | ips |
|---|---|---|---|
|fastspeech2_mix |语音合成声学模型|388MB|135 sequences/sec|
|hifigan_csmsc|语音合成声码器|873MB|30 sequences/sec|

## 2. 推理 Benchmark

参考 [TTS-Benchmark](https://github.com/PaddlePaddle/PaddleSpeech/wiki/TTS-Benchmark)。

## 3. 相关使用说明
