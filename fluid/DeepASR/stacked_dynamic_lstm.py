from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import time

import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.profiler as profiler
import data_utils.load_data as load_data

def parse_args():
    parser = argparse.ArgumentParser("LSTM model benchmark.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='The sequence number of a batch data. (default: %(default)d)')
    parser.add_argument(
        '--stacked_num',
        type=int,
        default=5,
        help='Number of lstm layers to stack. (default: %(default)d)')
    parser.add_argument(
        '--proj_dim',
        type=int,
        default=512,
        help='Project size of lstm unit. (default: %(default)d)')
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=1024,
        help='Hidden size of lstm unit. (default: %(default)d)')
    parser.add_argument(
        '--pass_num',
        type=int,
        default=100,
        help='Epoch number to train. (default: %(default)d)')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.002,
        help='Learning rate used to train. (default: %(default)f)')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type. (default: %(default)s)')
    parser.add_argument(
        '--infer_only', action='store_true', help='If set, run forward only.')
    parser.add_argument(
        '--use_cprof', action='store_true', help='If set, use cProfile.')
    parser.add_argument(
        '--use_nvprof',
        action='store_true',
        help='If set, use nvprof for CUDA.')
    args = parser.parse_args()
    return args


def print_arguments(args):
    vars(args)['use_nvprof'] = (vars(args)['use_nvprof'] and
                                vars(args)['device'] == 'GPU')
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def dynamic_lstm_model(hidden_dim,
                       stacked_num,
                       class_num=1749,
                       is_train=True):
    word_idx = fluid.layers.data(
            #name="feature", shape=[-1, 40], dtype="float32", lod_level=1)
            name="feature", shape=[-1, 120], dtype="float32", lod_level=1)

    seq_conv = fluid.layers.sequence_conv(input=word_idx, num_filters=1024, filter_size=3, filter_stride=1, bias_attr=True)
    #lstmp = seq_conv
    #for i in range(stacked_num):
    #    lstmp = fluid.layers.dynamic_lstmp(input=seq_conv, 4 * hidden_dim, hidden_dim)
    #    lstmp = fluid.layers.
        
    prediction = fluid.layers.fc(input=seq_conv, size=class_num, act='softmax')

    if not is_train: return word_idx, prediction

    label = fluid.layers.data(name="label", shape=[-1, 1], dtype="int64", lod_level=1)
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    return prediction, label, avg_cost

def dynamic_lstmp_model(hidden_dim,
                        proj_dim,
                        stacked_num,
                        class_num=1749,
                        is_train=True):
    feature = fluid.layers.data(
            #name="feature", shape=[-1, 40], dtype="float32", lod_level=1)
            name="feature", shape=[-1, 120], dtype="float32", lod_level=1)

    seq_conv1 = fluid.layers.sequence_conv(
                        input=feature, 
                        num_filters=1024, 
                        filter_size=3, 
                        filter_stride=1, 
                        bias_attr=True)
    bn1 = fluid.layers.batch_norm(
                        input=seq_conv1,
                        act="sigmoid",
                        is_test=False,
                        momentum=0.9,
                        epsilon=1e-05,
                        data_layout='NCHW')

    stack_input = bn1
    for i in range(stacked_num):
        #bias?????
        fc = fluid.layers.fc(input=stack_input, size=hidden_dim * 4, bias_attr=True)
        proj, cell = fluid.layers.dynamic_lstmp(
                            input=fc, 
                            size=hidden_dim * 4, 
                            proj_size=proj_dim, 
                            bias_attr=True,
                            use_peepholes=True,
                            is_reverse=False,
                            cell_activation="tanh",
                            proj_activation="tanh")
        bn = fluid.layers.batch_norm(
                            input=proj,
                            act="sigmoid",
                            is_test=False,
                            momentum=0.9,
                            epsilon=1e-05,
                            data_layout='NCHW')
        stack_input = bn
        
    prediction = fluid.layers.fc(input=stack_input, size=class_num, act='softmax')

    if not is_train: return feature, prediction

    label = fluid.layers.data(name="label", shape=[-1, 1], dtype="int64", lod_level=1)
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    return prediction, label, avg_cost

def train(args):
    if args.use_cprof:
        pr = cProfile.Profile()
        pr.enable()
    
    #word_dict = paddle.dataset.imdb.word_dict()
    #dict_size = len(word_dict)

    prediction, label, avg_cost = dynamic_lstmp_model(
        args.hidden_dim, args.proj_dim, args.stacked_num)

    adam_optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
    adam_optimizer.minimize(avg_cost)

    accuracy = fluid.evaluator.Accuracy(input=prediction, label=label)

    # clone from default main program
    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        test_accuracy = fluid.evaluator.Accuracy(input=prediction, label=label)
        test_target = [avg_cost] + test_accuracy.metrics + test_accuracy.states
        inference_program = fluid.io.get_inference_program(test_target)

    #train_reader = paddle.batch(
    #    paddle.reader.shuffle(
    #        paddle.dataset.imdb.train(word_dict), buf_size=25000),
    #    batch_size=args.batch_size)

    #test_reader = paddle.batch(
    #    paddle.reader.shuffle(
    #        paddle.dataset.imdb.test(word_dict), buf_size=25000),
    #    batch_size=args.batch_size)

    place = fluid.CPUPlace() if args.device == 'CPU' else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    #def do_validation():
    #    test_accuracy.reset(exe)

    #    for data in test_reader():
    #        loss, acc = exe.run(inference_program,
    #                            feed=feeder.feed(data),
    #                            fetch_list=[avg_cost] + test_accuracy.metrics)

    #    return test_accuracy.eval(exe)
    load_data.load_list()
    res_feature = fluid.LoDTensor()
    res_label = fluid.LoDTensor()
    for pass_id in xrange(args.pass_num):
        pass_start_time = time.time()
        words_seen = 0
        accuracy.reset(exe)
        batch_id = 0
        while True:
            # load_data
            one_batch = load_data.get_one_batch(args.batch_size)
            if one_batch == None:
                break
            (bat_feature, bat_label, lod) = one_batch
            res_feature.set(bat_feature, place)
            res_feature.set_lod([lod])
            res_label.set(bat_label, place)
            res_label.set_lod([lod])
            #print (bat_feature.shape)
            #print (bat_label.shape)
            #print (lod)

            batch_id += 1

            words_seen += lod[-1] 

            loss, acc = exe.run(fluid.default_main_program(),
                            feed={"feature":res_feature, "label":res_label},
                            fetch_list=[avg_cost] + accuracy.metrics,
                            return_numpy=False)
            train_acc = accuracy.eval(exe)
            print("acc:", lodtensor_to_ndarray(loss))
            #print(bat_label)

            #print("pass_id=%d, batch_id=%d, loss: %f, acc: %f, avg_acc: %f" %
            #      (pass_id, batch_id, loss, acc, train_acc))

        pass_end_time = time.time()
        time_consumed = pass_end_time - pass_start_time
        words_per_sec = words_seen / time_consumed
        #test_acc = do_validation()
        #print("pass_id=%d, test_acc: %f, words/s: %f, sec/pass: %f" %
        #      (pass_id, test_acc, words_per_sec, time_consumed))

def lodtensor_to_ndarray(lod_tensor):
    dims = lod_tensor.get_dims()
    ret = np.zeros(shape=dims).astype('float32')
    for i in xrange(np.product(dims)):
        ret.ravel()[i] = lod_tensor.get_float_element(i)
    return ret, lod_tensor.lod()

if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)

    if args.infer_only:
        pass
    else:
        if args.use_nvprof and args.device == 'GPU':
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                train(args)
        else:
            train(args)
