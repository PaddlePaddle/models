# __Deep Attention Matching Network__

This is the source code of Deep Attention Matching network (DAM), that is proposed for multi-turn response selection in the retrieval-based chatbot.

DAM is a neural matching network that entirely based on attention mechanism. The motivation of DAM is to capture those semantic dependencies, among dialogue elements at different level of granularities, in multi-turn conversation as matching evidences, in order to better match response candidate with its multi-turn context. DAM appears on ACL-2018, please find our paper at [http://aclweb.org/anthology/P18-1103](http://aclweb.org/anthology/P18-1103).


## __Network__

DAM is inspired by Transformer in Machine Translation (Vaswani et al., 2017), and we extend the key attention mechanism of Transformer in two perspectives and introduce those two kinds of attention in one uniform neural network.

- **self-attention** To gradually capture semantic representations in different granularities by stacking attention from word-level embeddings. Those multi-grained semantic representations would facilitate exploring segmental dependencies between context and response.

- **cross-attention** Attention across context and response can generally capture the relevance in dependency between segment pairs, which could provide complementary information to textual relevance for matching response with multi-turn context.

<p align="center">
<img src="images/Figure1.png"/> <br />
Overview of Deep Attention Matching Network
</p>

## __Results__

We test DAM on two large-scale multi-turn response selection tasks, i.e., the Ubuntu Corpus v1 and Douban Conversation Corpus, experimental results are bellow:

<p align="center">
<img src="images/Figure2.png"/> <br />
</p>

## __Usage__

Take the experiment on the Ubuntu Corpus v1 for Example.

1) Go to the `ubuntu` directory

```
cd ubuntu
```
2) Download the well-preprocessed data for training  

```
sh download_data.sh
```
3) Execute the model training and evaluation by

```
sh train.sh
```
for more detailed explanation about the arguments, please run

```
python ../train_and_evaluate.py --help
```

By default, the training is executed on one single GPU, which can be switched to multiple-GPU mode easily by simply resetting the visible devices in `train.sh`, e.g.,

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

4) Run test by

```
sh test.sh
```
and run the test for different saved models by using different argument `--model_path`.

Similary, one can carry out the experiment on the Douban Conversation Corpus by going to the directory `douban` and following the same procedure.

## __Dependencies__

- Python >= 2.7.3
- PaddlePaddle latest develop branch

## __Citation__

The following article describe the DAM in detail. We recommend citing this article as default.

```
@inproceedings{ ,
  title={Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network},
  author={Xiangyang Zhou, Lu Li, Daxiang Dong, Yi Liu, Ying Chen, Wayne Xin Zhao, Dianhai Yu and Hua Wu},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  volume={1},
  pages={  --  },
  year={2018}
}
```
