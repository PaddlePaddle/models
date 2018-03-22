#!/usr/bin/python
#encoding=utf8

import os
import sys
import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import config as conf
import reader
import utils
import time
import numpy as np

def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = core.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res

def rnn_lm(vocab_dim,
       emb_dim,
       hidden_dim):
    print "vocab_dim:%d emb_dim:%d hidden_dim:%d" % (vocab_dim, emb_dim, hidden_dim)
    src_seq = fluid.layers.data(name="src_seq", shape=[1], dtype="int64", lod_level=1)
    dst_seq = fluid.layers.data(name="dst_seq", shape=[1], dtype="int64", lod_level=1)
    input_emb = fluid.layers.embedding(input=src_seq, size=[vocab_dim, emb_dim])
    forward_proj = fluid.layers.fc(input=input_emb, size=hidden_dim * 4, bias_attr=True)
    forward, _ = fluid.layers.dynamic_lstm(
            input=forward_proj,
            size=hidden_dim * 4,
            use_peepholes=False);
    prediction = fluid.layers.fc(input=forward, size=vocab_dim, act='softmax', bias_attr=True)
    cost = fluid.layers.cross_entropy(input=prediction, label=dst_seq)
    avg_cost = fluid.layers.mean(x=cost)
    return src_seq, dst_seq, prediction, avg_cost

def main():
    # prepare vocab
    word_dict = paddle.dataset.imikolov.build_dict()
    dict_size = len(word_dict)
    vocab_dim = len(word_dict)
    utils.logger.info("dictionay size = %d" % vocab_dim)

    src_seq, dst_seq, prediction, avg_cost = rnn_lm(vocab_dim, conf.emb_dim, conf.hidden_size);
    optimizer = fluid.optimizer.SGD(learning_rate=0.1)
    optimizer.minimize(avg_cost)

    # evaluator
    batch_size = fluid.layers.create_tensor(dtype='int64')

    # define reader
    reader_args = {
        "file_name": conf.train_file,
        "word_dict": word_dict,
    }
    train_data = paddle.batch(
        paddle.dataset.imikolov.train(word_dict, 0, paddle.dataset.imikolov.DataType.SEQ),
        batch_size=conf.batch_size)
    test_reader = None
    if os.path.exists(conf.test_file) and os.path.getsize(conf.test_file):
        test_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.rnn_reader(**reader_args), buf_size=65536),
            batch_size=conf.batch_size)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    total_time = 0
    for pass_id in xrange(conf.num_passes):
        start_time = time.time()
        batch_id = 0
        for data in train_data():
            lod_src_seq = to_lodtensor(map(lambda x: x[0], data), place)
            lod_dst_seq = to_lodtensor(map(lambda x: x[1], data), place)
            outs = exe.run(
                fluid.default_main_program(),
                feed={"src_seq": lod_src_seq, "dst_seq": lod_dst_seq},
                fetch_list=[avg_cost])
            avg_cost_val = np.array(outs[0])
            if batch_id % conf.log_period == 0:
                print("Pass id: %d, batch id: %d, avg_cost: %f" %
                      (pass_id, batch_id, avg_cost_val))
            batch_id += 1
        end_time = time.time()
        total_time += (end_time - start_time)
    print("Total train time: %f" % (total_time))

if __name__ == "__main__":
    main()
