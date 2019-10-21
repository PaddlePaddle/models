# PLATO
**PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable**
[paper link](http://arxiv.org/abs/1910.07931)

## Requirements
```
- python >= 3.6
- paddlepaddle >= 1.5.2
- numpy
- nltk
- tqdm
- visualdl >= 1.3.0 (optional)
```

## Pre-trained dialogue generation model
A novel pre-training model for dialogue generation is introduced in this work, incorporated with latent discrete variables for one-to-many relationship modeling. Our model is flexible enough to support various kinds of conversations, including chit-chat, knowledge grounded dialogues, and conversational question answering. The pre-training is carried out with Reddit and Twitter corpora. You can download the uncased pre-trained model from:   
* PLATO, uncased [model](https://baidu-nlp.bj.bcebos.com/PLATO/model.tar.gz): 12-layers, 768-hidden, 12-heads, 132M parameters

```bash
mv /path/to/model.tar.gz .
tar xzf model.tar.gz
```

## Fine-tuning
We also provide instructions to fine-tune PLATO on different conversation datasets (chit-chat, knowledge grounded dialogues and conversational question answering).

### Data preparation
Download data from the [link](https://baidu-nlp.bj.bcebos.com/PLATO/data.tar.gz).
The tar file contains three processed datasets: DailyDialog, PersonaChat and DSTC7_AVSD.
```bash
mv /path/to/data.tar.gz .
tar xzf data.tar.gz
```

### Data format
Our model supports two kinds of data formats for dialogue context: "multi" and "multi_knowledge".
* multi: multi-turn dialogue context.
```txt
u_1 __eou__ u_2 __eou__ ... u_n \t r
```
* multi_knowledge: multi-turn dialogue context with background knowledge.
```txt
k_1 __eou__ k_2 __eou__ ... k_m \t u_1 __eou__ u_2 __eou__ ... u_n \t r
```

If you want to use this model on other datasets, you can process your data accordingly.

### Train
Fine-tuning the pre-trained model on different ${DATASET}.
```bash
# DailyDialog / PersonaChat / DSTC7_AVSD
DATASET=DailyDialog
sh scripts/${DATASET}/train.sh
```
After training, you can find the output folder `outputs/${DATASET}` (by default). It contatins `best.model` (best results on validation dataset), `hparams.json` (hyper-parameters of training script) and `trainer.log` (training log).

#### Recommended settings

For the fine-tuning of our pre-trained model, it usually requires about 10 epochs to reach convergence with learning rate = 1e-5 and about 2-3 epochs to reach convergence with learning rate = 5e-5.

GPU_MEM | batch_size | max_len
------|------|------
16G | 6 | 256
32G | 12 | 256

### Infer
Running inference on test dataset.
```bash
# DailyDialog / PersonaChat / DSTC7_AVSD
DATASET=DailyDialog
sh scripts/${DATASET}/infer.sh
```
After inference, you can find the output foler `outputs/${DATASET}.infer` (by default). It contains `infer_0.result.json` (the inference result), `hparams.json` (hyper-parameters of inference scipt) and `trainer.log` (inference log).

## Result

### DailyDialog
Model | BLEU-1/2 | Distinct-1/2 | Fluency | Coherence | Informativeness | Overall
------|------|------|------|------|------|-------
Seq2Seq | 0.336/0.268 | 0.030/0.128 | 1.85 | 0.37 | 0.44 | 0.33
iVAE_MI | 0.309/0.249 | 0.029/0.250 | 1.53 | 0.34 | 0.59 | 0.30
Our w/o Latent | 0.405/0.322 | 0.046/0.246 | 1.91 | 1.58 | 1.03 | 1.44
Our Method | 0.352/0.275 | 0.045/0.253 | 1.97 | 1.57 | 1.23 | 1.48

### PersonaChat
Model | BLEU-1/2 | Distinct-1/2 | Knowledge R/P/F1 | Fluency | Coherence | Informativeness | Overall
------|------|------|------|------|------|-------|-------
Seq2Seq | 0.448/0.353 | 0.004/0.016 | 0.004/0.016/0.006 | 1.82 | 0.37 | 0.85 | 0.34
LIC | 0.405/0.320 | 0.019/0.113 | 0.042/0.154/0.064 | 1.95 | 1.34 | 1.09 | 1.29
Our w/o Latent | 0.458/0.357 | 0.012/0.064 | 0.085/0.263/0.125 | 1.98 | 1.36 | 1.04 | 1.30
Our Method | 0.418/0.324 | 0.014/0.081 | 0.162/0.542/0.242 | 1.99 | 1.51 | 1.70 | 1.50

### DSTC7_AVSD
Model | BELU-1 | BELU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGH-L | CIDEr
------|------|------|------|------|------|-------|-------
Baseline | 0.629 | 0.485 | 0.383 | 0.309 | 0.215 | 0.487 | 0.746
CMU | 0.718 | 0.584 | 0.478 | 0.394 | 0.267 | 0.563 | 1.094
Our Method | 0.784 | 0.637 | 0.525 | 0.435 | 0.286 | 0.596 | 1.209
Our Method Upper Bound | 0.925 | 0.843 | 0.767 | 0.689 | 0.361 | 0.731 | 1.716

Note: In the experiments on DSTC_AVSD, the response selection of our method is strengthened with an extra ranking step, which ranks the candidates according to the automatic scores and selects the top one as the final answer.

## Citation
If you find PLATO useful in your work, please cite the following Arxiv paper:
```
@article{bao2019plato,
    title={PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable},
    author={Bao, Siqi and He, Huang, Wang, Fan and Wu, Hua},
    journal={arXiv preprint arXiv:1910.07931},
    year={2019}
}
```
## Contact information
For help or issues using PLATO, please submit a GitHub issue.

For personal communication related to PLATO, please contact Siqi Bao (`baosiqi@baidu.com`), or Huang He (`hehuang@baidu.com`).
