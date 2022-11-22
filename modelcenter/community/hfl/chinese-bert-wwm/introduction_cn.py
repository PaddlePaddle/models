#!/usr/bin/env python
# coding: utf-8

# ## Chinese BERT with Whole Word Masking
# For further accelerating Chinese natural language processing, we provide **Chinese pre-trained BERT with Whole Word Masking**.
# 
# **[Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/abs/1906.08101)**
# Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Ziqing Yang, Shijin Wang, Guoping Hu
# 
# This repository is developed based on：https://github.com/google-research/bert
# 
# You may also interested in,
# - Chinese BERT series: https://github.com/ymcui/Chinese-BERT-wwm
# - Chinese MacBERT: https://github.com/ymcui/MacBERT
# - Chinese ELECTRA: https://github.com/ymcui/Chinese-ELECTRA
# - Chinese XLNet: https://github.com/ymcui/Chinese-XLNet
# - Knowledge Distillation Toolkit - TextBrewer: https://github.com/airaria/TextBrewer
# 
# More resources by HFL: https://github.com/ymcui/HFL-Anthology
# 

# ## How to Use

# In[ ]:


get_ipython().system('pip install --upgrade paddlenlp')


# In[ ]:


import paddle
from paddlenlp.transformers import AutoModel

model = AutoModel.from_pretrained("hfl/chinese-bert-wwm")
input_ids = paddle.randint(100, 200, shape=[1, 20])
print(model(input_ids))


# 
# ## Citation
# If you find the technical report or resource is useful, please cite the following technical report in your paper.
# - Primary: https://arxiv.org/abs/2004.13922

# @inproceedings{cui-etal-2020-revisiting,
# title = "Revisiting Pre-Trained Models for {C}hinese Natural Language Processing",
# author = "Cui, Yiming  and
# Che, Wanxiang  and
# Liu, Ting  and
# Qin, Bing  and
# Wang, Shijin  and
# Hu, Guoping",
# booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings",
# month = nov,
# year = "2020",
# address = "Online",
# publisher = "Association for Computational Linguistics",
# url = "https://www.aclweb.org/anthology/2020.findings-emnlp.58",
# pages = "657--668",
# }
# 

# - Secondary: https://arxiv.org/abs/1906.08101
# 

# @article{chinese-bert-wwm,
# title={Pre-Training with Whole Word Masking for Chinese BERT},
# author={Cui, Yiming and Che, Wanxiang and Liu, Ting and Qin, Bing and Yang, Ziqing and Wang, Shijin and Hu, Guoping},
# journal={arXiv preprint arXiv:1906.08101},
# year={2019}
# }
# 

# > 模型来源 [hfl/chinese-bert-wwm](https://huggingface.co/hfl/chinese-bert-wwm)
