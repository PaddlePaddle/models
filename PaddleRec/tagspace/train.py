import os
import sys
import time
import six
import numpy as np
import math
import argparse
import paddle
import paddle.fluid as fluid
import time
import utils
import net

SEED = 102


def parse_args():
    parser = argparse.ArgumentParser("TagSpace benchmark.")
    parser.add_argument(
        '--neg_size', type=int, default=3, help='number of neg item')
    parser.add_argument(
        '--train_dir', type=str, default='train_data', help='train file')
    parser.add_argument(
        '--vocab_text_path', type=str, default='vocab_text.txt', help='text')
    parser.add_argument(
        '--vocab_tag_path', type=str, default='vocab_tag.txt', help='tag')
    parser.add_argument(
        '--model_dir', type=str, default='model_', help='model dir')
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
    train_dir = args.train_dir
    vocab_text_path = args.vocab_text_path
    vocab_tag_path = args.vocab_tag_path
    use_cuda = True if args.use_cuda else False
    parallel = True if args.parallel else False
    batch_size = args.batch_size
    neg_size = args.neg_size
    print("use_cuda: {}, parallel: {}, batch_size: {}, neg_size: {} "
          .format(use_cuda, parallel, batch_size, neg_size))
    vocab_text_size, vocab_tag_size, train_reader = utils.prepare_data(
        file_dir=train_dir,
        vocab_text_path=vocab_text_path,
        vocab_tag_path=vocab_tag_path,
        neg_size=neg_size,
        batch_size=batch_size * get_cards(args),
        buffer_size=batch_size * 100,
        is_train=True)
    """ train network """
    # Train program
    avg_cost, correct, cos_pos = net.network(
        vocab_text_size, vocab_tag_size, neg_size=neg_size)

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
    total_time = 0.0
    ce_info = []
    for pass_idx in range(pass_num):
        epoch_idx = pass_idx + 1
        print("epoch_%d start" % epoch_idx)
        t0 = time.time()
        for batch_id, data in enumerate(train_reader()):
            lod_text_seq = utils.to_lodtensor([dat[0] for dat in data], place)
            lod_pos_tag = utils.to_lodtensor([dat[1] for dat in data], place)
            lod_neg_tag = utils.to_lodtensor([dat[2] for dat in data], place)
            loss_val, correct_val = train_exe.run(
                feed={
                    "text": lod_text_seq,
                    "pos_tag": lod_pos_tag,
                    "neg_tag": lod_neg_tag
                },
                fetch_list=[avg_cost.name, correct.name])
            ce_info.append(
                float(np.sum(correct_val)) / (args.num_devices * batch_size))
            if batch_id % args.print_batch == 0:
                print("TRAIN --> pass: {} batch_num: {} avg_cost: {}, acc: {}"
                      .format(pass_idx, (batch_id + 10) * batch_size,
                              np.mean(loss_val),
                              float(np.sum(correct_val)) / (args.num_devices *
                                                            batch_size)))
        t1 = time.time()
        total_time += t1 - t0
        print("epoch:%d num_steps:%d time_cost(s):%f" %
              (epoch_idx, batch_id, total_time / epoch_idx))
        save_dir = "%s/epoch_%d" % (model_dir, epoch_idx)
        feed_var_names = ["text", "pos_tag"]
        fetch_vars = [cos_pos]
        fluid.io.save_inference_model(save_dir, feed_var_names, fetch_vars, exe)
    # only for ce
    if args.enable_ce:
        ce_acc = 0
        try:
            ce_acc = ce_info[-2]
        except:
            logger.error("ce info error")
        epoch_idx = args.pass_num
        device = get_device(args)
        if args.use_cuda:
            gpu_num = device[1]
            print("kpis\teach_pass_duration_gpu%s\t%s" %
                  (gpu_num, total_time / epoch_idx))
            print("kpis\ttrain_acc_gpu%s\t%s" % (gpu_num, ce_acc))
        else:
            cpu_num = device[1]
            threads_num = device[2]
            print("kpis\teach_pass_duration_cpu%s_thread%s\t%s" %
                  (cpu_num, threads_num, total_time / epoch_idx))
            print("kpis\ttrain_acc_cpu%s_thread%s\t%s" %
                  (cpu_num, threads_num, ce_acc))

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
