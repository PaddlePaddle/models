import os
import sys
import time
import six
import numpy as np
import math
import argparse
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers.nn as nn
import paddle.fluid.layers.tensor as tensor
import paddle.fluid.layers.control_flow as cf
import paddle.fluid.layers.io as io
import time
import utils

SEED = 102

def parse_args():
    parser = argparse.ArgumentParser("TagSpace benchmark.")
    parser.add_argument('train_file')
    parser.add_argument('test_file')
    parser.add_argument('--use_cuda', help='whether use gpu')
    args = parser.parse_args()
    return args

def network(vocab_text_size, vocab_tag_size, emb_dim=10, hid_dim=1000, win_size=5, margin=0.1, neg_size=5):
    """ network definition """
    text = io.data(name="text", shape=[1], lod_level=1, dtype='int64')
    pos_tag = io.data(name="pos_tag", shape=[1], lod_level=1, dtype='int64')
    neg_tag = io.data(name="neg_tag", shape=[1], lod_level=1, dtype='int64')
    
    text_emb = nn.embedding(
            input=text, size=[vocab_text_size, emb_dim], param_attr="text_emb")
    pos_tag_emb = nn.embedding(
            input=pos_tag, size=[vocab_tag_size, emb_dim], param_attr="tag_emb")
    neg_tag_emb = nn.embedding(
            input=neg_tag, size=[vocab_tag_size, emb_dim], param_attr="tag_emb")

    conv_1d = fluid.nets.sequence_conv_pool(
            input=text_emb,
            num_filters=hid_dim,
            filter_size=win_size,
            act="tanh",
            pool_type="max",
            param_attr="cnn")
    text_hid = fluid.layers.fc(input=conv_1d, size=emb_dim, param_attr="text_hid")
    cos_pos = nn.cos_sim(pos_tag_emb, text_hid)
    mul_text_hid = fluid.layers.sequence_expand_as(x=text_hid, y=neg_tag_emb)
    mul_cos_neg = nn.cos_sim(neg_tag_emb, mul_text_hid)
    cos_neg_all = fluid.layers.sequence_reshape(input=mul_cos_neg, new_dim=neg_size)
    #choose max negtive cosine
    cos_neg = nn.reduce_max(cos_neg_all, dim=1, keep_dim=True)
    #calculate hinge loss 
    loss_part1 = nn.elementwise_sub(
            tensor.fill_constant_batch_size_like(
                input=cos_pos,
                shape=[-1, 1],
                value=margin,
                dtype='float32'),
            cos_pos)
    loss_part2 = nn.elementwise_add(loss_part1, cos_neg)
    loss_part3 = nn.elementwise_max(
            tensor.fill_constant_batch_size_like(
                input=loss_part2, shape=[-1, 1], value=0.0, dtype='float32'),
            loss_part2)
    avg_cost = nn.mean(loss_part3)
    less = tensor.cast(cf.less_than(cos_neg, cos_pos), dtype='float32')
    correct = nn.reduce_sum(less)
    return text, pos_tag, neg_tag, avg_cost, correct, cos_pos

def train(train_reader, vocab_text, vocab_tag, base_lr, batch_size, neg_size,
          pass_num, use_cuda, model_dir):
    """ train network """
    args = parse_args()
    vocab_text_size = len(vocab_text)
    vocab_tag_size = len(vocab_tag)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    # Train program
    text, pos_tag, neg_tag, avg_cost, correct, cos_pos = network(vocab_text_size, vocab_tag_size, neg_size=neg_size)

    # Optimization to minimize lost
    sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=base_lr)
    sgd_optimizer.minimize(avg_cost)

    # Initialize executor
    startup_program = fluid.default_startup_program()
    loop_program = fluid.default_main_program()

    exe = fluid.Executor(place)
    exe.run(startup_program)

    total_time = 0.0
    for pass_idx in range(pass_num):
        epoch_idx = pass_idx + 1
        print("epoch_%d start" % epoch_idx)
        t0 = time.time()
        for batch_id, data in enumerate(train_reader()):
            lod_text_seq = utils.to_lodtensor([dat[0] for dat in data], place)
            lod_pos_tag = utils.to_lodtensor([dat[1] for dat in data], place)
            lod_neg_tag = utils.to_lodtensor([dat[2] for dat in data], place)
            loss_val, correct_val = exe.run(
                    loop_program,
                    feed={
                        "text": lod_text_seq,
                        "pos_tag": lod_pos_tag,
                        "neg_tag": lod_neg_tag},
                    fetch_list=[avg_cost, correct])
            if batch_id % 10 == 0:
                print("TRAIN --> pass: {} batch_id: {} avg_cost: {}, acc: {}"
                        .format(pass_idx, batch_id, loss_val,
                                float(correct_val) / batch_size))
        t1 = time.time()
        total_time += t1 - t0
        print("epoch:%d num_steps:%d time_cost(s):%f" %
              (epoch_idx, batch_id, total_time / epoch_idx))
        save_dir = "%s/epoch_%d" % (model_dir, epoch_idx)
        feed_var_names = ["text", "pos_tag"]
        fetch_vars = [cos_pos]
        fluid.io.save_inference_model(save_dir, feed_var_names, fetch_vars, exe)
    print("finish training")

def train_net():
    """ do training """
    args = parse_args()
    train_file = args.train_file
    test_file = args.test_file
    use_cuda = True if args.use_cuda else False
    batch_size = 100
    neg_size = 3
    vocab_text, vocab_tag, train_reader, test_reader = utils.prepare_data(
        train_file, test_file, neg_size=neg_size, batch_size=batch_size, buffer_size=batch_size*100, word_freq_threshold=0)
    train(
        train_reader=train_reader,
        vocab_text=vocab_text,
        vocab_tag=vocab_tag,
        base_lr=0.01,
        batch_size=batch_size,
        neg_size=neg_size,
        pass_num=10,
        use_cuda=use_cuda,
        model_dir="model")


if __name__ == "__main__":
    train_net()
