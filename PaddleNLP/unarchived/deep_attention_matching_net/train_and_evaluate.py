import os
import six
import numpy as np
import time
import argparse
import multiprocessing
import paddle
import paddle.fluid as fluid
import utils.reader as reader
from utils.util import print_arguments, mkdir

try:
    import cPickle as pickle  #python 2
except ImportError as e:
    import pickle  #python 3

from model import Net


#yapf: disable
def parse_args():
    parser = argparse.ArgumentParser("Training DAM.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for training. (default: %(default)d)')
    parser.add_argument(
        '--num_scan_data',
        type=int,
        default=2,
        help='Number of pass for training. (default: %(default)d)')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate used to train. (default: %(default)f)')
    parser.add_argument(
        '--data_path',
        type=str,
        default="data/data_small.pkl",
        help='Path to training data. (default: %(default)s)')
    parser.add_argument(
        '--save_path',
        type=str,
        default="saved_models",
        help='Path to save trained models. (default: %(default)s)')
    parser.add_argument(
        '--use_cuda',
        action='store_true',
        help='If set, use cuda for training.')
    parser.add_argument(
        '--use_pyreader',
        action='store_true',
        help='If set, use pyreader for reading data.')
    parser.add_argument(
        '--ext_eval',
        action='store_true',
        help='If set, use MAP, MRR ect for evaluation.')
    parser.add_argument(
        '--max_turn_num',
        type=int,
        default=9,
        help='Maximum number of utterances in context.')
    parser.add_argument(
        '--max_turn_len',
        type=int,
        default=50,
        help='Maximum length of setences in turns.')
    parser.add_argument(
        '--word_emb_init',
        type=str,
        default=None,
        help='Path to the initial word embedding.')
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=434512,
        help='The size of vocabulary.')
    parser.add_argument(
        '--emb_size',
        type=int,
        default=200,
        help='The dimension of word embedding.')
    parser.add_argument(
        '--_EOS_',
        type=int,
        default=28270,
        help='The id for the end of sentence in vocabulary.')
    parser.add_argument(
        '--stack_num',
        type=int,
        default=5,
        help='The number of stacked attentive modules in network.')
    parser.add_argument(
        '--channel1_num',
        type=int,
        default=32,
        help="The channels' number of the 1st conv3d layer's output.")
    parser.add_argument(
        '--channel2_num',
        type=int,
        default=16,
        help="The channels' number of the 2nd conv3d layer's output.")
    args = parser.parse_args()
    return args


#yapf: enable


def evaluate(score_path, result_file_path):
    if args.ext_eval:
        import utils.douban_evaluation as eva
    else:
        import utils.evaluation as eva
    #write evaluation result
    result = eva.evaluate(score_path)
    with open(result_file_path, 'w') as out_file:
        for p_at in result:
            out_file.write(str(p_at) + '\n')
    print('finish evaluation')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def test_with_feed(exe, program, feed_names, fetch_list, score_path, batches,
                   batch_num, dev_count):
    score_file = open(score_path, 'w')
    for it in six.moves.xrange(batch_num // dev_count):
        feed_list = []
        for dev in six.moves.xrange(dev_count):
            val_index = it * dev_count + dev
            batch_data = reader.make_one_batch_input(batches, val_index)
            feed_dict = dict(zip(feed_names, batch_data))
            feed_list.append(feed_dict)

            predicts = exe.run(feed=feed_list, fetch_list=fetch_list)

            scores = np.array(predicts[0])
            for dev in six.moves.xrange(dev_count):
                val_index = it * dev_count + dev
                for i in six.moves.xrange(args.batch_size):
                    score_file.write(
                        str(scores[args.batch_size * dev + i][0]) + '\t' + str(
                            batches["label"][val_index][i]) + '\n')
    score_file.close()


def test_with_pyreader(exe, program, pyreader, fetch_list, score_path, batches,
                       batch_num, dev_count):
    def data_provider():
        for index in six.moves.xrange(batch_num):
            yield reader.make_one_batch_input(batches, index)

    score_file = open(score_path, 'w')
    pyreader.decorate_tensor_provider(data_provider)
    it = 0
    pyreader.start()
    while True:
        try:
            predicts = exe.run(fetch_list=fetch_list)

            scores = np.array(predicts[0])
            for dev in six.moves.xrange(dev_count):
                val_index = it * dev_count + dev
                for i in six.moves.xrange(args.batch_size):
                    score_file.write(
                        str(scores[args.batch_size * dev + i][0]) + '\t' + str(
                            batches["label"][val_index][i]) + '\n')
            it += 1
        except fluid.core.EOFException:
            pyreader.reset()
            break
    score_file.close()


def train(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # data data_config
    data_conf = {
        "batch_size": args.batch_size,
        "max_turn_num": args.max_turn_num,
        "max_turn_len": args.max_turn_len,
        "_EOS_": args._EOS_,
    }

    dam = Net(args.max_turn_num, args.max_turn_len, args.vocab_size,
              args.emb_size, args.stack_num, args.channel1_num,
              args.channel2_num)

    train_program = fluid.Program()
    train_startup = fluid.Program()
    if "CE_MODE_X" in os.environ:
        train_program.random_seed = 110
        train_startup.random_seed = 110
    with fluid.program_guard(train_program, train_startup):
        with fluid.unique_name.guard():
            if args.use_pyreader:
                train_pyreader = dam.create_py_reader(
                    capacity=10, name='train_reader')
            else:
                dam.create_data_layers()
            loss, logits = dam.create_network()
            loss.persistable = True
            logits.persistable = True
            # gradient clipping
            fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByValue(
                max=1.0, min=-1.0))

            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.exponential_decay(
                    learning_rate=args.learning_rate,
                    decay_steps=400,
                    decay_rate=0.9,
                    staircase=True))
            optimizer.minimize(loss)
            print("begin memory optimization ...")
            fluid.memory_optimize(train_program)
            print("end memory optimization ...")

    test_program = fluid.Program()
    test_startup = fluid.Program()
    if "CE_MODE_X" in os.environ:
        test_program.random_seed = 110
        test_startup.random_seed = 110
    with fluid.program_guard(test_program, test_startup):
        with fluid.unique_name.guard():
            if args.use_pyreader:
                test_pyreader = dam.create_py_reader(
                    capacity=10, name='test_reader')
            else:
                dam.create_data_layers()

            loss, logits = dam.create_network()
            loss.persistable = True
            logits.persistable = True

    test_program = test_program.clone(for_test=True)

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    print("device count %d" % dev_count)
    print("theoretical memory usage: ")
    print(fluid.contrib.memory_usage(
        program=train_program, batch_size=args.batch_size))

    exe = fluid.Executor(place)
    exe.run(train_startup)
    exe.run(test_startup)

    train_exe = fluid.ParallelExecutor(
        use_cuda=args.use_cuda, loss_name=loss.name, main_program=train_program)

    test_exe = fluid.ParallelExecutor(
        use_cuda=args.use_cuda,
        main_program=test_program,
        share_vars_from=train_exe)

    if args.word_emb_init is not None:
        print("start loading word embedding init ...")
        if six.PY2:
            word_emb = np.array(pickle.load(open(args.word_emb_init,
                                                 'rb'))).astype('float32')
        else:
            word_emb = np.array(
                pickle.load(
                    open(args.word_emb_init, 'rb'), encoding="bytes")).astype(
                        'float32')
        dam.set_word_embedding(word_emb, place)
        print("finish init word embedding  ...")

    print("start loading data ...")
    with open(args.data_path, 'rb') as f:
        if six.PY2:
            train_data, val_data, test_data = pickle.load(f)
        else:
            train_data, val_data, test_data = pickle.load(f, encoding="bytes")
    print("finish loading data ...")

    val_batches = reader.build_batches(val_data, data_conf)

    batch_num = len(train_data[six.b('y')]) // args.batch_size
    val_batch_num = len(val_batches["response"])

    print_step = max(1, batch_num // (dev_count * 100))
    save_step = max(1, batch_num // (dev_count * 10))

    print("begin model training ...")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # train on one epoch data by feeding
    def train_with_feed(step):
        ave_cost = 0.0
        for it in six.moves.xrange(batch_num // dev_count):
            feed_list = []
            for dev in six.moves.xrange(dev_count):
                index = it * dev_count + dev
                batch_data = reader.make_one_batch_input(train_batches, index)
                feed_dict = dict(zip(dam.get_feed_names(), batch_data))
                feed_list.append(feed_dict)

            cost = train_exe.run(feed=feed_list, fetch_list=[loss.name])

            ave_cost += np.array(cost[0]).mean()
            step = step + 1
            if step % print_step == 0:
                print("processed: [" + str(step * dev_count * 1.0 / batch_num) +
                      "] ave loss: [" + str(ave_cost / print_step) + "]")
                ave_cost = 0.0

            if (args.save_path is not None) and (step % save_step == 0):
                save_path = os.path.join(args.save_path, "step_" + str(step))
                print("Save model at step %d ... " % step)
                print(time.strftime('%Y-%m-%d %H:%M:%S',
                                    time.localtime(time.time())))
                fluid.io.save_persistables(exe, save_path, train_program)

                score_path = os.path.join(args.save_path, 'score.' + str(step))
                test_with_feed(test_exe, test_program,
                               dam.get_feed_names(), [logits.name], score_path,
                               val_batches, val_batch_num, dev_count)

                result_file_path = os.path.join(args.save_path,
                                                'result.' + str(step))
                evaluate(score_path, result_file_path)
        return step, np.array(cost[0]).mean()

    # train on one epoch with pyreader
    def train_with_pyreader(step):
        def data_provider():
            for index in six.moves.xrange(batch_num):
                yield reader.make_one_batch_input(train_batches, index)

        train_pyreader.decorate_tensor_provider(data_provider)

        ave_cost = 0.0
        train_pyreader.start()
        while True:
            try:
                cost = train_exe.run(fetch_list=[loss.name])

                ave_cost += np.array(cost[0]).mean()
                step = step + 1
                if step % print_step == 0:
                    print("processed: [" + str(step * dev_count * 1.0 /
                                               batch_num) + "] ave loss: [" +
                          str(ave_cost / print_step) + "]")
                    ave_cost = 0.0

                if (args.save_path is not None) and (step % save_step == 0):
                    save_path = os.path.join(args.save_path,
                                             "step_" + str(step))
                    print("Save model at step %d ... " % step)
                    print(time.strftime('%Y-%m-%d %H:%M:%S',
                                        time.localtime(time.time())))
                    fluid.io.save_persistables(exe, save_path, train_program)

                    score_path = os.path.join(args.save_path,
                                              'score.' + str(step))
                    test_with_pyreader(test_exe, test_program, test_pyreader,
                                       [logits.name], score_path, val_batches,
                                       val_batch_num, dev_count)

                    result_file_path = os.path.join(args.save_path,
                                                    'result.' + str(step))
                    evaluate(score_path, result_file_path)

            except fluid.core.EOFException:
                train_pyreader.reset()
                break
        return step, np.array(cost[0]).mean()

    # train over different epoches
    global_step, train_time = 0, 0.0
    for epoch in six.moves.xrange(args.num_scan_data):
        shuffle_train = reader.unison_shuffle(
            train_data, seed=110 if ("CE_MODE_X" in os.environ) else None)
        train_batches = reader.build_batches(shuffle_train, data_conf)

        begin_time = time.time()
        if args.use_pyreader:
            global_step, last_cost = train_with_pyreader(global_step)
        else:
            global_step, last_cost = train_with_feed(global_step)

        pass_time_cost = time.time() - begin_time
        train_time += pass_time_cost
        print("Pass {0}, pass_time_cost {1}"
              .format(epoch, "%2.2f sec" % pass_time_cost))
    # For internal continuous evaluation
    if "CE_MODE_X" in os.environ:
        print("kpis	train_cost	%f" % last_cost)
        print("kpis	train_duration	%f" % train_time)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    train(args)
