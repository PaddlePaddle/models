import argparse
import sys
import time
import math
import unittest
import contextlib
import numpy as np
import six
import paddle.fluid as fluid
import paddle
import net
import utils
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf-8')


def parse_args():
    parser = argparse.ArgumentParser("PaddlePaddle Word2vec infer example")
    parser.add_argument(
        '--dict_path',
        type=str,
        default='./data/data_c/1-billion_dict_word_to_id_',
        help="The path of dic")
    parser.add_argument(
        '--infer_epoch',
        action='store_true',
        required=False,
        default=False,
        help='infer by epoch')
    parser.add_argument(
        '--infer_step',
        action='store_true',
        required=False,
        default=False,
        help='infer by step')
    parser.add_argument(
        '--test_dir', type=str, default='test_data', help='test file address')
    parser.add_argument(
        '--print_step', type=int, default='500000', help='print step')
    parser.add_argument(
        '--start_index', type=int, default='0', help='start index')
    parser.add_argument(
        '--start_batch', type=int, default='1', help='start index')
    parser.add_argument(
        '--end_batch', type=int, default='13', help='start index')
    parser.add_argument(
        '--last_index', type=int, default='100', help='last index')
    parser.add_argument(
        '--model_dir', type=str, default='model', help='model dir')
    parser.add_argument(
        '--use_cuda', type=int, default='0', help='whether use cuda')
    parser.add_argument(
        '--batch_size', type=int, default='5', help='batch_size')
    parser.add_argument('--emb_size', type=int, default='64', help='batch_size')
    args = parser.parse_args()
    return args


def infer_epoch(args, vocab_size, test_reader, use_cuda, i2w):
    """ inference function """
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    emb_size = args.emb_size
    batch_size = args.batch_size
    with fluid.scope_guard(fluid.Scope()):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            values, pred = net.infer_network(vocab_size, emb_size)
            for epoch in range(start_index, last_index + 1):
                copy_program = main_program.clone()
                model_path = model_dir + "/pass-" + str(epoch)
                fluid.load(copy_program, model_path, exe)
                accum_num = 0
                accum_num_sum = 0.0
                t0 = time.time()
                step_id = 0
                for data in test_reader():
                    step_id += 1
                    b_size = len([dat[0] for dat in data])
                    wa = np.array([dat[0] for dat in data]).astype(
                        "int64").reshape(b_size)
                    wb = np.array([dat[1] for dat in data]).astype(
                        "int64").reshape(b_size)
                    wc = np.array([dat[2] for dat in data]).astype(
                        "int64").reshape(b_size)

                    label = [dat[3] for dat in data]
                    input_word = [dat[4] for dat in data]
                    para = exe.run(copy_program,
                                   feed={
                                       "analogy_a": wa,
                                       "analogy_b": wb,
                                       "analogy_c": wc,
                                       "all_label": np.arange(vocab_size)
                                       .reshape(vocab_size).astype("int64"),
                                   },
                                   fetch_list=[pred.name, values],
                                   return_numpy=False)
                    pre = np.array(para[0])
                    val = np.array(para[1])
                    for ii in range(len(label)):
                        top4 = pre[ii]
                        accum_num_sum += 1
                        for idx in top4:
                            if int(idx) in input_word[ii]:
                                continue
                            if int(idx) == int(label[ii][0]):
                                accum_num += 1
                            break
                    if step_id % 1 == 0:
                        print("step:%d %d " % (step_id, accum_num))

                print("epoch:%d \t acc:%.3f " %
                      (epoch, 1.0 * accum_num / accum_num_sum))


def infer_step(args, vocab_size, test_reader, use_cuda, i2w):
    """ inference function """
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    emb_size = args.emb_size
    batch_size = args.batch_size
    with fluid.scope_guard(fluid.Scope()):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            values, pred = net.infer_network(vocab_size, emb_size)
            for epoch in range(start_index, last_index + 1):
                for batchid in range(args.start_batch, args.end_batch):
                    copy_program = main_program.clone()
                    model_path = model_dir + "/pass-" + str(epoch) + (
                        '/batch-' + str(batchid * args.print_step))
                    fluid.load(copy_program, model_path, exe)
                    accum_num = 0
                    accum_num_sum = 0.0
                    t0 = time.time()
                    step_id = 0
                    for data in test_reader():
                        step_id += 1
                        b_size = len([dat[0] for dat in data])
                        wa = np.array([dat[0] for dat in data]).astype(
                            "int64").reshape(b_size)
                        wb = np.array([dat[1] for dat in data]).astype(
                            "int64").reshape(b_size)
                        wc = np.array([dat[2] for dat in data]).astype(
                            "int64").reshape(b_size)

                        label = [dat[3] for dat in data]
                        input_word = [dat[4] for dat in data]
                        para = exe.run(
                            copy_program,
                            feed={
                                "analogy_a": wa,
                                "analogy_b": wb,
                                "analogy_c": wc,
                                "all_label":
                                np.arange(vocab_size).reshape(vocab_size),
                            },
                            fetch_list=[pred.name, values],
                            return_numpy=False)
                        pre = np.array(para[0])
                        val = np.array(para[1])
                        for ii in range(len(label)):
                            top4 = pre[ii]
                            accum_num_sum += 1
                            for idx in top4:
                                if int(idx) in input_word[ii]:
                                    continue
                                if int(idx) == int(label[ii][0]):
                                    accum_num += 1
                                break
                        if step_id % 1 == 0:
                            print("step:%d %d " % (step_id, accum_num))
                    print("epoch:%d \t acc:%.3f " %
                          (epoch, 1.0 * accum_num / accum_num_sum))
                    t1 = time.time()


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    utils.check_version()
    args = parse_args()
    start_index = args.start_index
    last_index = args.last_index
    test_dir = args.test_dir
    model_dir = args.model_dir
    batch_size = args.batch_size
    dict_path = args.dict_path
    use_cuda = True if args.use_cuda else False
    print("start index: ", start_index, " last_index:", last_index)
    vocab_size, test_reader, id2word = utils.prepare_data(
        test_dir, dict_path, batch_size=batch_size)
    print("vocab_size:", vocab_size)
    if args.infer_step:
        infer_step(
            args,
            vocab_size,
            test_reader=test_reader,
            use_cuda=use_cuda,
            i2w=id2word)
    else:
        infer_epoch(
            args,
            vocab_size,
            test_reader=test_reader,
            use_cuda=use_cuda,
            i2w=id2word)
