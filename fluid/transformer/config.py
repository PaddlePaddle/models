class TrainTaskConfig(object):
    use_gpu = False
    # the epoch number to train.
    pass_num = 2

    # number of sequences contained in a mini-batch.
    batch_size = 64

    # the hyper params for Adam optimizer.
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.98
    eps = 1e-9


class ModelHyperParams(object):
    # Dictionary size for source and target language. This model directly uses
    # paddle.dataset.wmt16 in which <bos>, <eos> and <unk> token has
    # alreay been added, but the <pad> token is not added. Transformer requires
    # sequences in a mini-batch are padded to have the same length. A <pad> token is
    # added into the original dictionary in paddle.dateset.wmt16.

    # size of source word dictionary.
    src_vocab_size = 10000
    # index for <pad> token in source language.
    src_pad_idx = src_vocab_size

    # size of target word dictionay
    trg_vocab_size = 10000
    # index for <pad> token in target language.
    trg_pad_idx = trg_vocab_size

    # position value corresponding to the <pad> token.
    pos_pad_idx = 0

    # max length of sequences. It should plus 1 to include position
    # padding token for position encoding.
    max_length = 50

    # the dimension for word embeddings, which is also the last dimension of
    # the input and output of multi-head attention, position-wise feed-forward
    # networks, encoder and decoder.

    d_model = 512
    # size of the hidden layer in position-wise feed-forward networks.
    d_inner_hid = 1024
    # the dimension that keys are projected to for dot-product attention.
    d_key = 64
    # the dimension that values are projected to for dot-product attention.
    d_value = 64
    # number of head used in multi-head attention.
    n_head = 8
    # number of sub-layers to be stacked in the encoder and decoder.
    n_layer = 6
    # dropout rate used by all dropout layers.
    dropout = 0.1


# Names of position encoding table which will be initialized externally.
pos_enc_param_names = (
    "src_pos_enc_table",
    "trg_pos_enc_table", )

# Names of all data layers listed in order.
input_data_names = (
    "src_word",
    "src_pos",
    "trg_word",
    "trg_pos",
    "src_slf_attn_bias",
    "trg_slf_attn_bias",
    "trg_src_attn_bias",
    "lbl_word", )
