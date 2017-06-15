# coding=utf-8

# -- config : data --

train_file = 'data/chinese.train.txt'
test_file = 'data/chinese.test.txt'
vocab_file = 'data/vocab_cn.txt'  # the file to save vocab

build_vocab_method = 'fixed_size'  # 'frequency' or 'fixed_size'
vocab_max_size = 3000  # when build_vocab_method = 'fixed_size'
unk_threshold = 1  # # when build_vocab_method = 'frequency'

min_sentence_length = 3
max_sentence_length = 60

# -- config : train --

use_which_model = 'ngram'  # must be: 'rnn' or 'ngram'
use_gpu = False  # whether to use gpu
trainer_count = 1  # number of trainer


class Config_rnn(object):
    """
    config for RNN language model
    """
    rnn_type = 'gru'  # or 'lstm'
    emb_dim = 200
    hidden_size = 200
    num_layer = 2
    num_passs = 2
    batch_size = 32
    model_file_name_prefix = 'lm_' + rnn_type + '_params_pass_'


class Config_ngram(object):
    """
    config for N-Gram language model
    """
    emb_dim = 200
    hidden_size = 200
    num_layer = 2
    N = 5
    num_passs = 2
    batch_size = 32
    model_file_name_prefix = 'lm_ngram_pass_'


# -- config : infer --

input_file = 'data/input.txt'  # input file contains sentence prefix each line
output_file = 'data/output.txt'  # the file to save results
num_words = 10  # the max number of words need to generate
beam_size = 5  # beam_width, the number of the prediction sentence for each prefix
