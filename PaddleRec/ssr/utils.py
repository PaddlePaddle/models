import numpy as np
import reader as reader
import os
import logging
import paddle.fluid as fluid
import paddle
import io


def get_vocab_size(vocab_path):
    with io.open(vocab_path, "r", encoding='utf-8') as rf:
        line = rf.readline()
        return int(line.strip())


def construct_train_data(file_dir, vocab_path, batch_size):
    vocab_size = get_vocab_size(vocab_path)
    files = [file_dir + '/' + f for f in os.listdir(file_dir)]
    y_data = reader.YoochooseDataset(vocab_size)
    train_reader = fluid.io.batch(
        paddle.reader.shuffle(
            y_data.train(files), buf_size=batch_size * 100),
        batch_size=batch_size)
    return train_reader, vocab_size


def construct_test_data(file_dir, vocab_path, batch_size):
    vocab_size = get_vocab_size(vocab_path)
    files = [file_dir + '/' + f for f in os.listdir(file_dir)]
    y_data = reader.YoochooseDataset(vocab_size)
    test_reader = fluid.io.batch(y_data.test(files), batch_size=batch_size)
    return test_reader, vocab_size


def check_version():
    """
     Log error and exit when the installed version of paddlepaddle is
     not satisfied.
     """
    err = "PaddlePaddle version 1.6 or higher is required, " \
          "or a suitable develop version is satisfied as well. \n" \
          "Please make sure the version is good with your code." \

    try:
        fluid.require_version('1.6.0')
    except Exception as e:
        logger.error(err)
        sys.exit(1)


def infer_data(raw_data, place):
    data = [dat[0] for dat in raw_data]
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
    p_label = [dat[1] for dat in raw_data]
    pos_label = np.array(p_label).astype("int64").reshape(len(p_label), 1)
    return res, pos_label
