"""
Reader for auto dialogue evaluation
"""

import sys
import time
import numpy as np
import random

import paddle.fluid as fluid
import paddle

def to_lodtensor(data, place):
    """
    Convert to LODtensor 
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


def reshape_batch(batch, place):
    """
    Reshape batch
    """
    context_reshape = to_lodtensor([dat[0] for dat in batch], place)
    response_reshape = to_lodtensor([dat[1] for dat in batch], place)
    label_reshape = [dat[2] for dat in batch]
    return (context_reshape, response_reshape, label_reshape)


def batch_reader(data_path,
                 batch_size,
                 place,
                 max_len=50,
                 sample_pro=1):
    """
    Yield batch
    """
    batch = []
    with open(data_path, 'r') as f:
        Print = True
        for line in f:
            #sample for training data
            if sample_pro < 1:
                if random.random() > sample_pro:
                    continue

            tokens = line.strip().split('\t')
            assert len(tokens) == 3
            context = [int(x) for x in tokens[0].split()[:max_len]]
            response = [int(x) for x in tokens[1].split()[:max_len]]

            label = [int(tokens[2])]
            #label = int(tokens[2])
            instance = (context, response, label)

            if len(batch) < batch_size:
                batch.append(instance)
            else:
                if len(batch) == batch_size:
                    yield reshape_batch(batch, place)
                batch = [instance]

        if len(batch) == batch_size:
            yield reshape_batch(batch, place)

