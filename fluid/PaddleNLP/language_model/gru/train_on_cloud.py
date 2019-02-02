import os
import sys
import time
import six

import numpy as np
import math

import collections
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework

cluster_train_dir = "./train/"
cluster_test_dir = "./test/"
train_file = "ptb.train.txt"
valid_file = "ptb.valid.txt"
test_file = "ptb.test.txt"


class DataType(object):
    """ data type """
    NGRAM = 1
    SEQ = 2


def word_count(f, word_freq=None):
    """ count words """
    if word_freq is None:
        word_freq = collections.defaultdict(int)

    for line in f:
        for w in line.strip().split():
            word_freq[w] += 1
        word_freq['<s>'] += 1
        word_freq['<e>'] += 1

    return word_freq


def build_dict(min_word_freq=50):
    """ build dictionary """
    train_filename = cluster_train_dir + train_file
    test_filename = cluster_test_dir + valid_file
    trainf = open(train_filename).readlines()
    testf = open(test_filename).readlines()
    word_freq = word_count(testf, word_count(trainf))
    if '<unk>' in word_freq:
        del word_freq['<unk>']
    word_freq = filter(lambda x: x[1] > min_word_freq, word_freq.items())
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*word_freq_sorted))
    word_idx = dict(zip(words, six.moves.xrange(len(words))))
    word_idx['<unk>'] = len(words)
    return word_idx


def reader_creator(filename, word_idx, n, data_type):
    """ create reader """

    def reader():
        if True:
            f = open(filename).readlines()
            UNK = word_idx['<unk>']
            for line in f:
                if DataType.NGRAM == data_type:
                    assert n > -1, 'Invalid gram length'
                    line = ['<s>'] + line.strip().split() + ['<e>']
                    if len(line) >= n:
                        line = [word_idx.get(w, UNK) for w in line]
                        for i in range(n, len(line) + 1):
                            yield tuple(line[i - n:i])
                elif DataType.SEQ == data_type:
                    line = line.strip().split()
                    line = [word_idx.get(w, UNK) for w in line]
                    src_seq = [word_idx['<s>']] + line
                    trg_seq = line + [word_idx['<e>']]
                    if n > 0 and len(src_seq) > n:
                        continue
                    yield src_seq, trg_seq
                else:
                    assert False, 'Unknow data type'

    return reader


def to_lodtensor(data, place):
    """ convert to LODtensor """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for line in seq_lens:
        cur_len += line
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def prepare_data(batch_size, buffer_size=1000, word_freq_threshold=0):
    """ prepare the English Pann Treebank (PTB) data """
    vocab = build_dict(word_freq_threshold)
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader_creator(
                cluster_train_dir + train_file,
                vocab,
                buffer_size,
                data_type=DataType.SEQ),
            buf_size=buffer_size),
        batch_size)
    test_reader = paddle.batch(
        reader_creator(
            cluster_test_dir + test_file,
            vocab,
            buffer_size,
            data_type=DataType.SEQ),
        batch_size)
    return vocab, train_reader, test_reader


def network(src, dst, vocab_size, hid_size, init_low_bound, init_high_bound):
    """ network definition """
    emb_lr_x = 10.0
    gru_lr_x = 1.0
    fc_lr_x = 1.0
    emb = fluid.layers.embedding(
        input=src,
        size=[vocab_size, hid_size],
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=init_low_bound, high=init_high_bound),
            learning_rate=emb_lr_x),
        is_sparse=True)

    fc0 = fluid.layers.fc(input=emb,
                          size=hid_size * 3,
                          param_attr=fluid.ParamAttr(
                              initializer=fluid.initializer.Uniform(
                                  low=init_low_bound, high=init_high_bound),
                              learning_rate=gru_lr_x))
    gru_h0 = fluid.layers.dynamic_gru(
        input=fc0,
        size=hid_size,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=init_low_bound, high=init_high_bound),
            learning_rate=gru_lr_x))

    fc = fluid.layers.fc(input=gru_h0,
                         size=vocab_size,
                         act='softmax',
                         param_attr=fluid.ParamAttr(
                             initializer=fluid.initializer.Uniform(
                                 low=init_low_bound, high=init_high_bound),
                             learning_rate=fc_lr_x))

    cost = fluid.layers.cross_entropy(input=fc, label=dst)
    return cost


def do_train(train_reader,
             vocab,
             network,
             hid_size,
             base_lr,
             batch_size,
             pass_num,
             use_cuda,
             parallel,
             model_dir,
             init_low_bound=-0.04,
             init_high_bound=0.04):
    """ train network """
    vocab_size = len(vocab)

    src_wordseq = fluid.layers.data(
        name="src_wordseq", shape=[1], dtype="int64", lod_level=1)
    dst_wordseq = fluid.layers.data(
        name="dst_wordseq", shape=[1], dtype="int64", lod_level=1)

    avg_cost = None
    if not parallel:
        cost = network(src_wordseq, dst_wordseq, vocab_size, hid_size,
                       init_low_bound, init_high_bound)
        avg_cost = fluid.layers.mean(x=cost)
    else:
        places = fluid.layers.device.get_places()
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            cost = network(
                pd.read_input(src_wordseq),
                pd.read_input(dst_wordseq), vocab_size, hid_size,
                init_low_bound, init_high_bound)
            pd.write_output(cost)

        cost = pd()
        avg_cost = fluid.layers.mean(x=cost)

    sgd_optimizer = fluid.optimizer.SGD(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=base_lr,
            decay_steps=2100 * 4,
            decay_rate=0.5,
            staircase=True))
    sgd_optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())
    total_time = 0.0
    for pass_idx in six.moves.xrange(pass_num):
        epoch_idx = pass_idx + 1
        print("epoch_%d start" % epoch_idx)

        t0 = time.time()
        i = 0
        for data in train_reader():
            i += 1
            lod_src_wordseq = to_lodtensor([dat[0] for dat in data], place)
            lod_dst_wordseq = to_lodtensor([dat[1] for dat in data], place)
            ret_avg_cost = exe.run(fluid.default_main_program(),
                                   feed={
                                       "src_wordseq": lod_src_wordseq,
                                       "dst_wordseq": lod_dst_wordseq
                                   },
                                   fetch_list=[avg_cost],
                                   use_program_cache=True)
            avg_ppl = math.exp(ret_avg_cost[0])
            if i % 100 == 0:
                print("step:%d ppl:%.3f" % (i, avg_ppl))

        t1 = time.time()
        total_time += t1 - t0
        print("epoch:%d num_steps:%d time_cost(s):%f" %
              (epoch_idx, i, total_time / epoch_idx))

        save_dir = "%s/epoch_%d" % (model_dir, epoch_idx)
        feed_var_names = ["src_wordseq", "dst_wordseq"]
        fetch_vars = [avg_cost]
        fluid.io.save_inference_model(save_dir, feed_var_names, fetch_vars, exe)
        print("model saved in %s" % save_dir)

    print("finish training")


def train():
    """ do training """
    batch_size = 20
    vocab, train_reader, test_reader = prepare_data(
        batch_size=batch_size, buffer_size=1000, word_freq_threshold=0)

    # End batch and end pass event handler
    def event_handler(event):
        """ event handler """
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print("\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            print("isinstance(event, paddle.event.EndPass)")

    do_train(
        train_reader=train_reader,
        vocab=vocab,
        network=network,
        hid_size=200,
        base_lr=1.0,
        batch_size=batch_size,
        pass_num=12,
        use_cuda=True,
        parallel=False,
        model_dir="./output/model",
        init_low_bound=-0.1,
        init_high_bound=0.1)


if __name__ == "__main__":
    if not os.path.exists("./output/model"):
        os.makedirs("./output/model")
    train()
