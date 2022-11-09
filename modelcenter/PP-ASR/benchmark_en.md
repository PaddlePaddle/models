## Inference benchmark

### 1.1 Software and hardware environment

* PP-ASR model inference speed test using single GPU V100, using batch size=1, using CUDA 10.2, CUDNN 7.5.1.

### 1.2 Datasets
PP-ASR model uses the train from wenetspeech as the training set and the test from aishell1 as the test set.

### 1.3 Performance

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


## 2. Relevant instructions
Please refer to: https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/examples/wenetspeech/asr1
