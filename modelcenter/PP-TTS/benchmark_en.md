## 1. Training Benchmark

### 1.1 Environment

* FastSpeech2，2 GPUs，batch size =  64 per GPU.
* HiFiGAN，1 GPU，GPU batch size = 16 per GPU.
* python version: 3.7.0
* paddle version: v2.4.0rc0
* machine: 8x Tesla V100-SXM2-32GB, 24 core Intel(R) Xeon(R) Gold 6148, 100Gbps RDMA network


### 1.2 Datasets

| language | dataset |audio info | describtion |
| -------- | -------- | -------- | -------- |
| Chinese | [CSMSC](https://www.data-baker.com/open_source.html) | 48KHz, 16bit | single speaker，female，12 h|
| Chinese | [AISHELL-3](http://www.aishelltech.com/aishell_3) | 44.1kHz，16bit |multi-speakers，85 h|
| English | [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/) | 22050Hz, 16bit | single speaker，female，24 h|
| English | [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) | 48kHz, 16bit | multi-speakers，44 h|

### 1.3 Benchmark

|model | task | model_size | ips |
|---|---|---|---|
|fastspeech2_mix |TTS Acoustic Model|388MB|135 sequences/sec|
|hifigan_csmsc|TTS Vocoder|873MB|30 sequences/sec|

## 2. Inference Benchmark

Please refer to [TTS-Benchmark](https://github.com/PaddlePaddle/PaddleSpeech/wiki/TTS-Benchmark).

## 3. Reference
