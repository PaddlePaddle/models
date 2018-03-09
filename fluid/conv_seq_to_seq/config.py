import copy
from utils import objdic

# special token id of WMT14 dataset.
end_id = 6
start_id = 0
unk_id = 2

train_config = objdic(
    mode='train',
    max_len=40,
    batch_size=10,
    # buffer for shuffling of data reader
    buf_size=1000,
    num_pass=1000,
    encoder=objdic(
        dict_size=30000,
        num_embeddings=30000 + 1,
        pad_id=30000,  # treat the last wordid as pad_id
        word_dim=512,
        convolutions=[[512, 3]] * 20,
        end_id=end_id,
        start_id=start_id,
        unk_id=unk_id,
    ),
    decoder=objdic(
        dict_size=30000,
        num_embeddings=30000 + 1,
        pad_id=30000,
        word_dim=512,
        beam_size=5,
        convolutions=[[512, 3]] * 20,
        end_id=end_id,
        start_id=start_id,
        unk_id=unk_id,
    ))

debug_train_config = objdic(
    mode='debug',
    max_len=40,
    batch_size=10,
    # buffer for shuffling of data reader
    buf_size=1000,
    num_pass=100,
    encoder=objdic(
        dict_size=30000,
        num_embeddings=30000 + 1,
        pad_id=30000,
        word_dim=128,
        convolutions=[[128, 3]] * 3,
        end_id=end_id,
        start_id=start_id,
        unk_id=unk_id,
    ),
    decoder=objdic(
        dict_size=30000,
        num_embeddings=30000 + 1,
        pad_id=30000,
        word_dim=128,
        beam_size=5,
        convolutions=[[128, 3]] * 3,
        end_id=end_id,
        start_id=start_id,
        unk_id=unk_id,
    ))
