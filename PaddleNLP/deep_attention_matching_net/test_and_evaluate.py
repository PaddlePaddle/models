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
    parser = argparse.ArgumentParser("Test for DAM.")
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
        default="data/ubuntu/data_small.pkl",
        help='Path to training data. (default: %(default)s)')
    parser.add_argument(
        '--save_path',
        type=str,
        default="./",
        help='Path to save score and result files. (default: %(default)s)')
    parser.add_argument(
        '--model_path',
        type=str,
        default="saved_models/step_1000",
        help='Path to load well-trained models. (default: %(default)s)')
    parser.add_argument(
        '--use_cuda',
        action='store_true',
        help='If set, use cuda for training.')
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
        help='The id for end of sentence in vocabulary.')
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


def test(args):
    if not os.path.exists(args.save_path):
        mkdir(args.save_path)
    if not os.path.exists(args.model_path):
        raise ValueError("Invalid model init path %s" % args.model_path)
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
    dam.create_data_layers()
    loss, logits = dam.create_network()

    loss.persistable = True
    logits.persistable = True

    # gradient clipping
    fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByValue(
        max=1.0, min=-1.0))

    test_program = fluid.default_main_program().clone(for_test=True)
    optimizer = fluid.optimizer.Adam(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=args.learning_rate,
            decay_steps=400,
            decay_rate=0.9,
            staircase=True))
    optimizer.minimize(loss)

    fluid.memory_optimize(fluid.default_main_program())

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = multiprocessing.cpu_count()

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    fluid.io.load_persistables(exe, args.model_path)

    test_exe = fluid.ParallelExecutor(
        use_cuda=args.use_cuda, main_program=test_program)

    print("start loading data ...")
    with open(args.data_path, 'rb') as f:
        if six.PY2:
            train_data, val_data, test_data = pickle.load(f)
        else:
            train_data, val_data, test_data = pickle.load(f, encoding="bytes")
    print("finish loading data ...")

    if args.ext_eval:
        import utils.douban_evaluation as eva
        eval_metrics = ["MAP", "MRR", "P@1", "R_{10}@1", "R_{10}@2", "R_{10}@5"]
    else:
        import utils.evaluation as eva
        eval_metrics = ["R_2@1", "R_{10}@1", "R_{10}@2", "R_{10}@5"]

    test_batches = reader.build_batches(test_data, data_conf)

    test_batch_num = len(test_batches["response"])

    print("test batch num: %d" % test_batch_num)

    print("begin inference ...")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    score_path = os.path.join(args.save_path, 'score.txt')
    score_file = open(score_path, 'w')

    for it in six.moves.xrange(test_batch_num // dev_count):
        feed_list = []
        for dev in six.moves.xrange(dev_count):
            index = it * dev_count + dev
            batch_data = reader.make_one_batch_input(test_batches, index)
            feed_dict = dict(zip(dam.get_feed_names(), batch_data))
            feed_list.append(feed_dict)

        predicts = test_exe.run(feed=feed_list, fetch_list=[logits.name])

        scores = np.array(predicts[0])
        print("step = %d" % it)

        for dev in six.moves.xrange(dev_count):
            index = it * dev_count + dev
            for i in six.moves.xrange(args.batch_size):
                score_file.write(
                    str(scores[args.batch_size * dev + i][0]) + '\t' + str(
                        test_batches["label"][index][i]) + '\n')

    score_file.close()

    #write evaluation result
    result = eva.evaluate(score_path)
    result_file_path = os.path.join(args.save_path, 'result.txt')
    with open(result_file_path, 'w') as out_file:
        for metric, p_at in zip(eval_metrics, result):
            out_file.write(metric + ": " + str(p_at) + '\n')
    print('finish test')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    test(args)
