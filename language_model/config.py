#!/usr/bin/env python
# coding=utf-8
import os

# -- config : building dictionary --
max_word_num = 5120 - 3
cutoff_word_fre = 10

# -- config : data --
train_file = "data/chinese.train.txt"
test_file = ""
vocab_file = "data/vocab_dict.txt"
batch_size = 4
num_passes = 1

model_save_dir = "models"
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)

# -- config : train --
model_type = "rnn"  # must be: "rnn" or "ngram"
use_gpu = False  # whether to use gpu
trainer_count = 1  # number of trainer


class ConfigRnn(object):
    """
    config for RNN language model
    """
    rnn_type = "lstm"  # "gru" or "lstm"
    emb_dim = 256
    hidden_size = 256
    num_layer = 2


class ConfigNgram(object):
    """
    config for N-Gram language model
    """
    emb_dim = 256
    hidden_size = 256
    num_layer = 2
    N = 5


# -- config : infer --
output_file = "data/output.txt"  # the file to save results
num_words = 50  # the max number of words need to generate
beam_size = 5  # beam_width, the number of the prediction sentence for each prefix
