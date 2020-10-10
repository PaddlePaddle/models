import os
import sys
import time
import six
import numpy as np
import math
import argparse
import paddle.fluid as fluid
import paddle
import time
import utils
import net

SEED = 102


def parse_args():
    parser = argparse.ArgumentParser("gru4rec benchmark.")
    parser.add_argument(
        '--train_dir', type=str, default='train_data', help='train file')
    parser.add_argument(
        '--vocab_path', type=str, default='vocab.txt', help='vocab file')
    parser.add_argument(
        '--is_local', type=int, default=1, help='whether is local')
    parser.add_argument(
        '--hid_size', type=int, default=100, help='hidden-dim size')
    parser.add_argument(
        '--model_dir', type=str, default='model_recall20', help='model dir')
    parser.add_argument(
        '--batch_size', type=int, default=5, help='num of batch size')
    parser.add_argument(
        '--print_batch', type=int, default=10, help='num of print batch')
    parser.add_argument(
        '--pass_num', type=int, default=10, help='number of epoch')
    parser.add_argument(
        '--use_cuda', type=int, default=0, help='whether use gpu')
    parser.add_argument(
        '--parallel', type=int, default=0, help='whether parallel')
    parser.add_argument(
        '--base_lr', type=float, default=0.01, help='learning rate')
    parser.add_argument(
        '--num_devices', type=int, default=1, help='Number of GPU devices')
    parser.add_argument(
        '--step_num', type=int, default=1000, help='Number of steps')
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help='If set, run the task with continuous evaluation logs.')
    args = parser.parse_args()
    return args


def get_cards(args):
    return args.num_devices


def train():
    """ do training """
    args = parse_args()
    if args.enable_ce:
        fluid.default_startup_program().random_seed = SEED
        fluid.default_main_program().random_seed = SEED
    hid_size = args.hid_size
    train_dir = args.train_dir
    vocab_path = args.vocab_path
    use_cuda = True if args.use_cuda else False
    parallel = True if args.parallel else False
    print("use_cuda:", use_cuda, "parallel:", parallel)
    batch_size = args.batch_size
    vocab_size, train_reader = utils.prepare_data(
        train_dir, vocab_path, batch_size=batch_size * get_cards(args),\
        buffer_size=1000, word_freq_threshold=0, is_train=True)

    # Train program
    src_wordseq, dst_wordseq, avg_cost, acc = net.all_vocab_network(
        vocab_size=vocab_size, hid_size=hid_size)

    # Optimization to minimize lost
    sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=args.base_lr)
    sgd_optimizer.minimize(avg_cost)

    # Initialize executor
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    if parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=use_cuda, loss_name=avg_cost.name)
    else:
        train_exe = exe

    pass_num = args.pass_num
    model_dir = args.model_dir
    fetch_list = [avg_cost.name]

    ce_info = []
    total_time = 0.0
    for pass_idx in six.moves.xrange(pass_num):
        epoch_idx = pass_idx + 1
        print("epoch_%d start" % epoch_idx)

        t0 = time.time()
        i = 0
        newest_ppl = 0
        for data in train_reader():
            i += 1
            lod_src_wordseq = utils.to_lodtensor([dat[0] for dat in data],
                                                 place)
            lod_dst_wordseq = utils.to_lodtensor([dat[1] for dat in data],
                                                 place)
            ret_avg_cost = train_exe.run(feed={
                "src_wordseq": lod_src_wordseq,
                "dst_wordseq": lod_dst_wordseq
            },
                                         fetch_list=fetch_list)
            avg_ppl = np.exp(ret_avg_cost[0])
            newest_ppl = np.mean(avg_ppl)
            ce_info.append(newest_ppl)
            if i % args.print_batch == 0:
                print("step:%d ppl:%.3f" % (i, newest_ppl))
            if args.enable_ce and i > args.step_num:
                break

        t1 = time.time()
        total_time += t1 - t0
        print("epoch:%d num_steps:%d time_cost(s):%f" %
              (epoch_idx, i, total_time / epoch_idx))
        save_dir = "%s/epoch_%d" % (model_dir, epoch_idx)
        feed_var_names = ["src_wordseq", "dst_wordseq"]
        fetch_vars = [avg_cost, acc]
        fluid.io.save_inference_model(save_dir, feed_var_names, fetch_vars, exe)
        print("model saved in %s" % save_dir)

    # only for ce
    if args.enable_ce:
        ce_ppl = 0
        try:
            ce_ppl = ce_info[-2]
        except:
            print("ce info error")
        epoch_idx = args.pass_num
        device = get_device(args)
        if args.use_cuda:
            gpu_num = device[1]
            print("kpis\teach_pass_duration_gpu%s\t%s" %
                  (gpu_num, total_time / epoch_idx))
            print("kpis\ttrain_ppl_gpu%s\t%s" % (gpu_num, ce_ppl))
        else:
            cpu_num = device[1]
            threads_num = device[2]
            print("kpis\teach_pass_duration_cpu%s_thread%s\t%s" %
                  (cpu_num, threads_num, total_time / epoch_idx))
            print("kpis\ttrain_ppl_cpu%s_thread%s\t%s" %
                  (cpu_num, threads_num, ce_ppl))

    print("finish training")


def get_device(args):
    if args.use_cuda:
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES", 1)
        gpu_num = len(gpus.split(','))
        return "gpu", gpu_num
    else:
        threads_num = os.environ.get('NUM_THREADS', 1)
        cpu_num = os.environ.get('CPU_NUM', 1)
        return "cpu", int(cpu_num), int(threads_num)


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    utils.check_version()
    train()
