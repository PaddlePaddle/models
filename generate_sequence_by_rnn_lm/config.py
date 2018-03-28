import os

################## for building word dictionary  ##################

max_word_num = 51200 - 2
cutoff_word_fre = 0

################## for training task  #########################
# path of training data
train_file = "data/train_data_examples.txt"
# path of testing data, if testing file does not exist,
# testing will not be performed at the end of each training pass
test_file = ""
# path of word dictionary, if this file does not exist,
# word dictionary will be built from training data.
vocab_file = "data/word_vocab.txt"
# directory to save the trained model
# create a new directory if the directoy does not exist
model_save_dir = "models"

batch_size = 32  # the number of training examples in one forward/backward pass
num_passes = 20  # how many passes to train the model

log_period = 50
save_period_by_batches = 50

use_gpu = False  # to use gpu or not
trainer_count = 1  # number of trainer

##################  for model configuration  ##################
rnn_type = "lstm"  # "gru" or "lstm"
emb_dim = 256
hidden_size = 256
stacked_rnn_num = 2

##################  for text generation  ##################
gen_file = "data/train_data_examples.txt"
gen_result = "data/gen_result.txt"
max_gen_len = 25  # the max number of words to generate
beam_size = 5
model_path = "models/rnn_lm_pass_00000.tar.gz"

if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
