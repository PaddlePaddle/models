## Conformer Steaming Pretrained Model

Pretrain model from https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar.gz

| Model | Params | Config | Augmentation| Test set | Decode method | Chunk Size | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| conformer | 32.52 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | attention | 16 | 0.056273 |  
| conformer | 32.52 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | ctc_greedy_search | 16 | 0.078918 |  
| conformer | 32.52 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | ctc_prefix_beam_search | 16 | 0.079080 |  
| conformer | 32.52 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | attention_rescoring | 16 | 0.054401 |

| Model | Params | Config | Augmentation| Test set | Decode method | Chunk Size | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| conformer | 32.52 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | attention | -1 | 0.050767 |  
| conformer | 32.52 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | ctc_greedy_search | -1 | 0.061884 |  
| conformer | 32.52 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | ctc_prefix_beam_search | -1 | 0.062056 |  
| conformer | 32.52 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | attention_rescoring | -1 |  0.052110 |
