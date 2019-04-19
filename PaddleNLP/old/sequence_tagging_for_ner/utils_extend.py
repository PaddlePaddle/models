import numpy as np

import paddle.fluid as fluid


def get_embedding(emb_file='data/wordVectors.txt'):
    """
    Get the trained word vector.
    """
    return np.loadtxt(emb_file, dtype='float32')


def to_lodtensor(data, place):
    """
    convert data to lodtensor
    """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res
