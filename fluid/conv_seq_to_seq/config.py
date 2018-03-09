import copy
from utils import objdic

train_config = objdic(
    max_len=40,
    batch_size=10,
    # buffer for shuffling of data reader
    buf_size=1000,
    num_pass=1000,
    encoder=objdic(
        dict_size=30000,
        word_dim=512,
        convolutions=[[512, 3]] * 20,
    ),
    decoder=objdic(
        dict_size=30000,
        word_dim=512,
        beam_size=5,
        convolutions=[[512, 3]] * 20,
    ))

debug_train_config = objdic(
    max_len=40,
    batch_size=10,
    # buffer for shuffling of data reader
    buf_size=1000,
    num_pass=1000,
    encoder=objdic(
        dict_size=30000,
        word_dim=128,
        convolutions=[[128, 3]] * 3,
    ),
    decoder=objdic(
        dict_size=30000,
        word_dim=128,
        beam_size=5,
        convolutions=[[128, 3]] * 3,
    ))
