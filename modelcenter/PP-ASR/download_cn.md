## 流式Conformer语音识别模型

预训练模型下载：https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar.gz

| 模型 | 参数量 |  数据增广 | 测试数据集 | 解码策略 | 流式块大小 | 字符错误率 |  
| --- | --- | --- | --- | --- | --- | --- |
| conformer | 32.52 M | spec_aug  | aishell1 | attention | 16 | 0.056273 |  
| conformer | 32.52 M | spec_aug  | aishell1 | ctc_greedy_search | 16 | 0.078918 |  
| conformer | 32.52 M | spec_aug  | aishell1 | ctc_prefix_beam_search | 16 | 0.079080 |  
| conformer | 32.52 M | spec_aug  | aishell1 | attention_rescoring | 16 | 0.054401 |

| 模型 | 参数量 |  数据增广 | 测试数据集 | 解码策略 | 流式块大小 | 字符错误率 |
| --- | --- | --- | --- | --- | --- | --- |
| conformer | 32.52 M |  spec_aug  | aishell1 | attention | -1 | 0.050767 |  
| conformer | 32.52 M |  spec_aug  | aishell1 | ctc_greedy_search | -1 | 0.061884 |  
| conformer | 32.52 M |  spec_aug  | aishell1 | ctc_prefix_beam_search | -1 | 0.062056 |  
| conformer | 32.52 M | spec_aug  | aishell1 | attention_rescoring | -1 |  0.052110 |
