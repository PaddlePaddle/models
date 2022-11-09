## 推理 Benchmark

### 1.1 软硬件环境

* PP-ASR 模型推理速度测试采用单卡V100，使用 batch size=1 进行测试，使用 CUDA 10.2, CUDNN 7.5.1

### 1.2 数据集
PP-ASR 模型使用 wenetspeech 数据集中的 train 作为训练集，使用 aishell1 中的 test 作为测试集.

### 1.3 指标

| Model | Decoding Method | Chunk Size | CER |  RTF |
| --- | --- | --- | --- | --- |
| conformer | attention | 16 | 0.056273 | 0.0003696 |
| conformer | ctc_greedy_search | 16 | 0.078918 |  0.0001571|
| conformer | ctc_prefix_beam_search | 16 | 0.079080 |  0.0002221 |
| conformer | attention_rescoring | 16 | 0.054401 | 0.0002569 |

| Model | Decoding Method | Chunk Size | CER |  RTF |
| --- | --- | --- | --- | --- |
| conformer |  attention | -1 | 0.050767 |  0.0003589 |
| conformer |  ctc_greedy_search | -1 | 0.061884 |  0.0000435 |
| conformer | ctc_prefix_beam_search | -1 | 0.062056 |  0.0001934|
| conformer | attention_rescoring | -1 |  0.052110 |0.0002103|


## 2. 相关使用说明
请参考：https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/examples/wenetspeech/asr1
