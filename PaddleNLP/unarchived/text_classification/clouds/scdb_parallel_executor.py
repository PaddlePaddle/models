import unittest
import contextlib
import paddle
import paddle.fluid as fluid
import numpy as np
import six
import sys
import time
import os
import json
import random


def to_lodtensor(data, place):
    """
    convert to LODtensor
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


def load_vocab(filename):
    """
    load imdb vocabulary
    """
    vocab = {}
    with open(filename) as f:
        wid = 0
        for line in f:
            vocab[line.strip()] = wid
            wid += 1
    vocab["<unk>"] = len(vocab)
    return vocab


def data2tensor(data, place):
    """
    data2tensor
    """
    input_seq = to_lodtensor([x[0] for x in data], place)
    y_data = np.array([x[1] for x in data]).astype("int64")
    y_data = y_data.reshape([-1, 1])
    return {"words": input_seq, "label": y_data}


def data2pred(data, place):
    """
    data2tensor
    """
    input_seq = to_lodtensor([x[0] for x in data], place)
    y_data = np.array([x[1] for x in data]).astype("int64")
    y_data = y_data.reshape([-1, 1])
    return {"words": input_seq}


def load_dict(vocab):
    """
    Load dict from vocab
    """
    word_dict = dict()
    with open(vocab, "r") as fin:
        for line in fin:
            cols = line.strip("\r\n").decode("gb18030").split("\t")
            word_dict[cols[0]] = int(cols[1])
    return word_dict


def save_dict(word_dict, vocab):
    """
    Save dict into file
    """
    with open(vocab, "w") as fout:
        for k, v in six.iteritems(word_dict):
            outstr = ("%s\t%s\n" % (k, v)).encode("gb18030")
            fout.write(outstr)


def build_dict(fname):
    """
    build word dict using trainset
    """
    word_dict = dict()
    with open(fname, "r") as fin:
        for line in fin:
            try:
                words = line.strip("\r\n").decode("gb18030").split("\t")[
                    1].split(" ")
            except:
                sys.stderr.write("[warning] build_dict: decode error\n")
                continue
            for w in words:
                if w not in word_dict:
                    word_dict[w] = len(word_dict)
    return word_dict


def scdb_word_dict(vocab="scdb_data/train_set/train.vocab"):
    """
    get word_dict
    """
    if not os.path.exists(vocab):
        w_dict = build_dict(train_file)
        save_dict(w_dict, vocab)
    else:
        w_dict = load_dict(vocab)
    w_dict["<unk>"] = len(w_dict)
    return w_dict


def data_reader(fname, word_dict, is_dir=False):
    """
    Convert word sequence into slot
    """
    unk_id = len(word_dict)
    all_data = []
    filelist = []
    if is_dir:
        filelist = [fname + os.sep + f for f in os.listdir(fname)]
    else:
        filelist = [fname]

    for each_name in filelist:
        with open(each_name, "r") as fin:
            for line in fin:
                try:
                    cols = line.strip("\r\n").decode("gb18030").split("\t")
                except:
                    sys.stderr.write("warning: ignore decode error\n")
                    continue

                label = int(cols[0])
                wids = [
                    word_dict[x] if x in word_dict else unk_id
                    for x in cols[1].split(" ")
                ]
                all_data.append((wids, label))

    random.shuffle(all_data)

    def reader():
        for doc, label in all_data:
            yield doc, label

    return reader


def scdb_train_data(train_dir="scdb_data/train_set/corpus.train.seg",
                    w_dict=None):
    """
    create train data
    """
    return data_reader(train_dir, w_dict, True)


def scdb_test_data(test_file, w_dict):
    """
    test_set=["car", "lbs", "spot", "weibo",
            "baby", "toutiao", "3c", "movie", "haogan"]
    """
    return data_reader(test_file, w_dict)


def bow_net(data,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2):
    """
    bow net
    """
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow)
    fc_1 = fluid.layers.fc(input=bow_tanh, size=hid_dim, act="tanh")
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim2, act="tanh")
    prediction = fluid.layers.fc(input=[fc_2], size=class_dim, act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, acc, prediction


def cnn_net(data,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2,
            win_size=3):
    """
    conv net
    """
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])

    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=win_size,
        act="tanh",
        pool_type="max")

    fc_1 = fluid.layers.fc(input=[conv_3], size=hid_dim2)

    prediction = fluid.layers.fc(input=[fc_1], size=class_dim, act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, acc, prediction


def lstm_net(data,
             label,
             dict_dim,
             emb_dim=128,
             hid_dim=128,
             hid_dim2=96,
             class_dim=2,
             emb_lr=30.0):
    """
    lstm net
    """
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(learning_rate=emb_lr))

    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)

    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0, size=hid_dim * 4, is_reverse=False)

    lstm_max = fluid.layers.sequence_pool(input=lstm_h, pool_type='max')
    lstm_max_tanh = fluid.layers.tanh(lstm_max)

    fc1 = fluid.layers.fc(input=lstm_max_tanh, size=hid_dim2, act='tanh')

    prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax')

    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, acc, prediction


def bilstm_net(data,
               label,
               dict_dim,
               emb_dim=128,
               hid_dim=128,
               hid_dim2=96,
               class_dim=2,
               emb_lr=30.0):
    """
    lstm net
    """
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(learning_rate=emb_lr))

    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)

    rfc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)

    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0, size=hid_dim * 4, is_reverse=False)

    rlstm_h, c = fluid.layers.dynamic_lstm(
        input=rfc0, size=hid_dim * 4, is_reverse=True)

    lstm_last = fluid.layers.sequence_last_step(input=lstm_h)
    rlstm_last = fluid.layers.sequence_last_step(input=rlstm_h)

    lstm_last_tanh = fluid.layers.tanh(lstm_last)
    rlstm_last_tanh = fluid.layers.tanh(rlstm_last)

    lstm_concat = fluid.layers.concat(input=[lstm_last, rlstm_last], axis=1)

    fc1 = fluid.layers.fc(input=lstm_concat, size=hid_dim2, act='tanh')

    prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax')

    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, acc, prediction


def gru_net(data,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2,
            emb_lr=30.0):
    """
    gru net
    """
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(learning_rate=emb_lr))

    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 3)

    gru_h = fluid.layers.dynamic_gru(input=fc0, size=hid_dim, is_reverse=False)

    gru_max = fluid.layers.sequence_pool(input=gru_h, pool_type='max')
    gru_max_tanh = fluid.layers.tanh(gru_max)

    fc1 = fluid.layers.fc(input=gru_max_tanh, size=hid_dim2, act='tanh')

    prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax')

    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, acc, prediction


def infer(test_reader, use_cuda, model_path=None):
    """
    inference function
    """
    if model_path is None:
        print(str(model_path) + " cannot be found")
        return

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(model_path, exe)

        class2_list, class3_list = [], []
        for each_test_reader in test_reader:
            class2_acc, class3_acc = 0.0, 0.0
            total_count, neu_count = 0, 0

            for data in each_test_reader():
                pred = exe.run(inference_program,
                               feed=data2pred(data, place),
                               fetch_list=fetch_targets,
                               return_numpy=True)

                for i, val in enumerate(data):
                    pos_score = pred[0][i, 1]
                    true_label = val[1]
                    if true_label == 2.0 and pos_score > 0.5:
                        class2_acc += 1
                    if true_label == 0.0 and pos_score < 0.5:
                        class2_acc += 1

                    if true_label == 2.0 and pos_score > 0.55:
                        class3_acc += 1
                    if true_label == 1.0 and pos_score > 0.45 and pos_score <= 0.55:
                        class3_acc += 1
                    if true_label == 0.0 and pos_score <= 0.45:
                        class3_acc += 1

                    if true_label == 1.0:
                        neu_count += 1

                total_count += len(data)

            class2_acc = class2_acc / (total_count - neu_count)
            class3_acc = class3_acc / total_count
            class2_list.append(class2_acc)
            class3_list.append(class3_acc)

        class2_acc = sum(class2_list) / len(class2_list)
        class3_acc = sum(class3_list) / len(class3_list)
        print("[test info] model_path: %s, class2_acc: %f, class3_acc: %f" %
              (model_path, class2_acc, class3_acc))


def start_train(train_reader,
                test_reader,
                word_dict,
                network,
                use_cuda,
                parallel,
                save_dirname,
                lr=0.2,
                batch_size=128,
                pass_num=30):
    """
    train network
    """
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)

    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    cost, acc, pred = network(data, label, len(word_dict) + 1)

    sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=lr)
    sgd_optimizer.minimize(cost)

    place = fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

    start_exe = fluid.Executor(place)
    start_exe.run(fluid.default_startup_program())

    exe = fluid.ParallelExecutor(use_cuda, loss_name=cost.name)
    for pass_id in six.moves.xrange(pass_num):
        total_acc, total_cost, total_count, avg_cost, avg_acc = 0.0, 0.0, 0.0, 0.0, 0.0
        for data in train_reader():
            cost_val, acc_val = exe.run(feed=feeder.feed(data),
                                        fetch_list=[cost.name, acc.name])
            cost_val_list, acc_val_list = np.array(cost_val), np.array(acc_val)
            total_cost += cost_val_list.sum() * len(data)
            total_acc += acc_val_list.sum() * len(data)
            total_count += len(data)

        avg_cost = total_cost / total_count
        avg_acc = total_acc / total_count
        print("[train info]: pass_id: %d, avg_acc: %f, avg_cost: %f" %
              (pass_id, avg_acc, avg_cost))

        gpu_place = fluid.CUDAPlace(0)
        save_exe = fluid.Executor(gpu_place)
        epoch_model = save_dirname + "/" + "epoch" + str(pass_id)
        fluid.io.save_inference_model(epoch_model, ["words"], pred, save_exe)
        infer(test_reader, False, epoch_model)


def train_net(vocab="./thirdparty/train.vocab",
              train_dir="./train",
              test_list=["car", "spot", "weibo", "lbs"]):
    """
    w_dict = scdb_word_dict(vocab=vocab)
    test_files = [ "./thirdparty" + os.sep + f for f in test_list]

    train_reader = paddle.batch(
                        scdb_train_data(train_dir, w_dict),
                        batch_size = 256)

    test_reader = [paddle.batch(scdb_test_data(test_file, w_dict), batch_size = 50) \
            for test_file in test_files]
    """
    w_dict = paddle.dataset.imdb.word_dict()
    print("dict ready")
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(w_dict), buf_size=50000),
        batch_size=128)

    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.test(w_dict), buf_size=50000),
        batch_size=128)
    test_reader = [test_reader]
    start_train(
        train_reader,
        test_reader,
        w_dict,
        bilstm_net,
        use_cuda=True,
        parallel=False,
        save_dirname="scdb_bilstm_model",
        lr=0.05,
        pass_num=10,
        batch_size=256)


if __name__ == "__main__":
    train_net()
