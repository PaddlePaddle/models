# Represent the dict sizes of source and target language. The dict from the
# dataset here used includes the <bos>, <eos> and <unk> token but exlcudes
# the <pad> token. It should plus 1 to include the padding token when used as
# the size of lookup table.
src_vocab_size = 10000
trg_vocab_size = 10000
# Represent the id of <pad> token in source language.
src_pad_idx = src_vocab_size
# Represent the id of <pad> token in target language.
trg_pad_idx = trg_vocab_size
# Represent the position value corresponding to the <pad> token.
pos_pad_idx = 0
# Represent the max length of sequences. It should plus 1 to include position
# padding token for position encoding.
max_length = 50
# Represent the epoch number to train.
pass_num = 2
# Represent the number of sequences contained in a mini-batch.
batch_size = 64
# Reprent the params for Adam optimizer.
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.98
eps = 1e-9
# Represent the dimension of embeddings, which is also the last dimension of
# the input and output of multi-head attention, position-wise feed-forward
# networks, encoder and decoder.
d_model = 512
# Represent the size of the hidden layer in position-wise feed-forward networks.
d_inner_hid = 1024
# Represent the dimension keys are projected to for dot-product attention.
d_key = 64
# Represent the dimension values are projected to for dot-product attention.
d_value = 64
# Represent the number of head used in multi-head attention.
n_head = 8
# Represent the number of sub-layers to be stacked in the encoder and decoder.
n_layer = 6
# Represent the dropout rate used by all dropout layers.
dropout = 0.1

# Names of position encoding table which will be initialized in external.
pos_enc_param_names = ("src_pos_enc_table", "trg_pos_enc_table")
# Names of all data layers listed in order.
input_data_names = ("src_word", "src_pos", "trg_word", "trg_pos",
                    "src_slf_attn_bias", "trg_slf_attn_bias",
                    "trg_src_attn_bias", "lbl_word")
