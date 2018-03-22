"""seq2seq model for fluid."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import time

import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor

from model import seq_to_seq_net
from config import TrainConfig as train_conf
from config import ModelConfig as model_conf


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    lod_t = core.LoDTensor()
    lod_t.set(flattened_data, place)
    lod_t.set_lod([lod])
    return lod_t, lod[-1]


def lodtensor_to_ndarray(lod_tensor):
    dims = lod_tensor.get_dims()
    ndarray = np.zeros(shape=dims).astype('float32')
    for i in xrange(np.product(dims)):
        ndarray.ravel()[i] = lod_tensor.get_float_element(i)
    return ndarray


def train():
    src_word_idx = fluid.layers.data(
        name='source_sequence', shape=[1], dtype='int64', lod_level=1)
    trg_word_idx = fluid.layers.data(
        name='target_sequence', shape=[1], dtype='int64', lod_level=1)
    label = fluid.layers.data(
        name='label_sequence', shape=[1], dtype='int64', lod_level=1)
    if train_conf.parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            src_word_idx_ = pd.read_input(src_word_idx)
            trg_word_idx_ = pd.read_input(trg_word_idx)
            label_ = pd.read_input(label)
            avg_cost = seq_to_seq_net(
                src_word_idx_,
                trg_word_idx_,
                label_,
                embedding_dim=model_conf.embedding_dim,
                encoder_size=model_conf.encoder_size,
                decoder_size=model_conf.decoder_size,
                source_dict_dim=model_conf.source_dict_dim,
                target_dict_dim=model_conf.target_dict_dim,
                is_generating=model_conf.is_generating,
                beam_size=model_conf.beam_size,
                max_length=model_conf.max_length)
            pd.write_output(avg_cost)
        avg_cost = pd()
        avg_cost = fluid.layers.mean(x=avg_cost)
    else:
        avg_cost = seq_to_seq_net(
            src_word_idx,
            trg_word_idx,
            label,
            embedding_dim=model_conf.embedding_dim,
            encoder_size=model_conf.encoder_size,
            decoder_size=model_conf.decoder_size,
            source_dict_dim=model_conf.source_dict_dim,
            target_dict_dim=model_conf.target_dict_dim,
            is_generating=model_conf.is_generating,
            beam_size=model_conf.beam_size,
            max_length=model_conf.max_length)

    feeding_list = ["source_sequence", "target_sequence", "label_sequence"]
    # clone from default main program
    inference_program = fluid.default_main_program().clone()

    optimizer = fluid.optimizer.Adam(learning_rate=train_conf.learning_rate)
    optimizer.minimize(avg_cost)

    train_batch_generator = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt16.train(model_conf.source_dict_dim,
                                       model_conf.target_dict_dim),
            buf_size=train_conf.buf_size),
        batch_size=train_conf.batch_size)

    test_batch_generator = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt16.test(model_conf.source_dict_dim,
                                      model_conf.target_dict_dim),
            buf_size=train_conf.buf_size),
        batch_size=train_conf.batch_size)

    place = core.CUDAPlace(0) if train_conf.use_gpu else core.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    def do_validation():
        total_loss = 0.0
        count = 0
        for batch_id, data in enumerate(test_batch_generator()):
            src_seq = to_lodtensor(map(lambda x: x[0], data), place)[0]
            trg_seq = to_lodtensor(map(lambda x: x[1], data), place)[0]
            lbl_seq = to_lodtensor(map(lambda x: x[2], data), place)[0]

            fetch_outs = exe.run(inference_program,
                                 feed={
                                     feeding_list[0]: src_seq,
                                     feeding_list[1]: trg_seq,
                                     feeding_list[2]: lbl_seq
                                 },
                                 fetch_list=[avg_cost],
                                 return_numpy=False)

            total_loss += lodtensor_to_ndarray(fetch_outs[0])[0]
            count += 1

        return total_loss / count

    for pass_id in xrange(train_conf.pass_num):
        pass_start_time = time.time()
        words_seen = 0
        for batch_id, data in enumerate(train_batch_generator()):
            src_seq, word_num = to_lodtensor(map(lambda x: x[0], data), place)
            words_seen += word_num
            trg_seq, word_num = to_lodtensor(map(lambda x: x[1], data), place)
            words_seen += word_num
            lbl_seq, _ = to_lodtensor(map(lambda x: x[2], data), place)

            fetch_outs = exe.run(framework.default_main_program(),
                                 feed={
                                     feeding_list[0]: src_seq,
                                     feeding_list[1]: trg_seq,
                                     feeding_list[2]: lbl_seq
                                 },
                                 fetch_list=[avg_cost])

            avg_cost_val = np.array(fetch_outs[0])
            print('pass_id=%d, batch_id=%d, train_loss: %f' %
                  (pass_id, batch_id, avg_cost_val))

        pass_end_time = time.time()
        test_loss = do_validation()
        time_consumed = pass_end_time - pass_start_time
        words_per_sec = words_seen / time_consumed
        print("pass_id=%d, test_loss: %f, words/s: %f, sec/pass: %f" %
              (pass_id, test_loss, words_per_sec, time_consumed))


def infer():
    pass


if __name__ == '__main__':
    if train_conf.infer_only:
        infer()
    else:
        train()
