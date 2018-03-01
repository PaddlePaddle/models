#!/usr/bin/python
#encoding=utf8

import os
import sys
import paddle.v2 as paddle
import paddle.fluid as fluid
import config as conf
import reader
import utils
import time

def rnn_lm(vocab_dim,
       emb_dim,
       hidden_dim):
    print "vocab_dim:%d emb_dim:%d hidden_dim:%d" % (vocab_dim, emb_dim, hidden_dim)
    data = fluid.layers.data(
        name="input", shape=[1], dtype="int64", lod_level=1)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64", lod_level=1)
    input_emb = fluid.layers.embedding(input=data, size=[vocab_dim, emb_dim])
    print input_emb
    # only support lstm here
    forward, _ = fluid.layers.dynamic_lstm(
            input=input_emb,
            size=hidden_dim * 4,
            use_peepholes=False);
    prediction = fluid.layers.fc(input=forward, size=vocab_dim, act='softmax')
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    return data, label, prediction, avg_cost

def main():
    model_save_dir="models"
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    # prepare vocab
    if not (os.path.exists(conf.vocab_file) and
            os.path.getsize(conf.vocab_file)):
        utils.logger.info(("word dictionary does not exist, "
                     "build it from the training data"))
        utils.build_dict(conf.train_file, conf.vocab_file, conf.max_word_num,
                   conf.cutoff_word_fre)
    utils.logger.info("load word dictionary.")
    word_dict = utils.load_dict(conf.vocab_file)
    vocab_dim = len(word_dict)
    utils.logger.info("dictionay size = %d" % vocab_dim)

    data, label, prediction, avg_cost = rnn_lm(vocab_dim, conf.emb_dim, conf.hidden_size);
    optimizer = fluid.optimizer.Adam(learning_rate=1e-3, beta1=0.9, beta2=0.98, epsilon=1e-9)
    optimizer.minimize(avg_cost)
    accuracy = fluid.evaluator.Accuracy(input=prediction, label=label)

    # define reader
    reader_args = {
        "file_name": conf.train_file,
        "word_dict": word_dict,
    }
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.rnn_reader(**reader_args), buf_size=102400),
        batch_size=conf.batch_size)
    test_reader = None
    if os.path.exists(conf.test_file) and os.path.getsize(conf.test_file):
        test_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.rnn_reader(**reader_args), buf_size=65536),
            batch_size=conf.batch_size)

    if conf.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)
    exe.run(fluid.default_startup_program())

    total_time = 0
    for pass_id in xrange(conf.num_passes):
        accuracy.reset(exe)
        start_time = time.time()
        for batch_id, data in enumerate(train_reader()):
        print "batch_id:" + str(batch_id)
            cost_val, acc_val = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_cost, accuracy.metrics[0]], return_numpy=False)
            pass_acc = accuracy.eval(exe)
            if batch_id % conf.log_period == 0:
        print pass_id
                print batch_id
                print type(cost_val)
                print type(pass_acc)
                print("Pass id: %d, batch id: %d, cost: %f, pass_acc %f" %
                      (pass_id, batch_id, utils.lodtensor_to_ndarray(cost_val)[0],
                      pass_acc[0]))
        end_time = time.time()
        total_time += (end_time - start_time)
    print("Total train time: %f" % (total_time))


if __name__ == "__main__":
    main()
