#Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import time
import argparse
import logging
import paddle.fluid as fluid
import paddle
import utils
import numpy as np
from nets import SequenceSemanticRetrieval

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser("sequence semantic retrieval")
    parser.add_argument(
        "--train_dir", type=str, default='train_data', help="Training file")
    parser.add_argument(
        "--base_lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        '--vocab_path', type=str, default='vocab.txt', help='vocab file')
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        '--parallel', type=int, default=0, help='whether parallel')
    parser.add_argument(
        '--use_cuda', type=int, default=0, help='whether use gpu')
    parser.add_argument(
        '--print_batch', type=int, default=10, help='num of print batch')
    parser.add_argument(
        '--model_dir', type=str, default='model_output', help='model dir')
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="hidden size")
    parser.add_argument(
        "--batch_size", type=int, default=50, help="number of batch")
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="embedding dim")
    parser.add_argument(
        '--num_devices', type=int, default=1, help='Number of GPU devices')
    parser.add_argument(
        '--step_num', type=int, default=1000, help='Number of steps')
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help='If set, run the task with continuous evaluation logs.')
    parser.add_argument(
        '--role', type=str, default='pserver', help='trainer or pserver')
    parser.add_argument(
        '--endpoints',
        type=str,
        default='127.0.0.1:6000',
        help='The pserver endpoints, like: 127.0.0.1:6000, 127.0.0.1:6001')
    parser.add_argument(
        '--current_endpoint',
        type=str,
        default='127.0.0.1:6000',
        help='The current_endpoint')
    parser.add_argument(
        '--trainer_id',
        type=int,
        default=0,
        help='trainer id ,only trainer_id=0 save model')
    parser.add_argument(
        '--trainers',
        type=int,
        default=1,
        help='The num of trianers, (default: 1)')
    return parser.parse_args()


def get_cards(args):
    return args.num_devices


def train_loop(main_program, avg_cost, acc, train_input_data, place, args,
               train_reader):
    data_list = [var.name for var in train_input_data]
    feeder = fluid.DataFeeder(feed_list=data_list, place=place)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    train_exe = exe

    total_time = 0.0
    ce_info = []
    for pass_id in range(args.epochs):
        epoch_idx = pass_id + 1
        print("epoch_%d start" % epoch_idx)
        t0 = time.time()
        i = 0
        for batch_id, data in enumerate(train_reader()):
            i += 1
            loss_val, correct_val = train_exe.run(
                feed=feeder.feed(data), fetch_list=[avg_cost.name, acc.name])
            ce_info.append(float(np.mean(correct_val)) / args.batch_size)
            if i % args.print_batch == 0:
                logger.info(
                    "Train --> pass: {} batch_id: {} avg_cost: {}, acc: {}".
                    format(pass_id, batch_id,
                           np.mean(loss_val),
                           float(np.mean(correct_val)) / args.batch_size))
            if args.enable_ce and i > args.step_num:
                break
        t1 = time.time()
        total_time += t1 - t0
        print("epoch:%d num_steps:%d time_cost(s):%f" %
              (epoch_idx, i, total_time / epoch_idx))
        save_dir = "%s/epoch_%d" % (args.model_dir, epoch_idx)
        fluid.io.save_params(executor=exe, dirname=save_dir)
        print("model saved in %s" % save_dir)

    # only for ce
    if args.enable_ce:
        ce_acc = 0
        try:
            ce_acc = ce_info[-2]
        except:
            print("ce info error")
        epoch_idx = args.epochs
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


def train(args):
    if args.enable_ce:
        SEED = 102
        fluid.default_startup_program().random_seed = SEED
        fluid.default_main_program().random_seed = SEED
    use_cuda = True if args.use_cuda else False
    parallel = True if args.parallel else False
    print("use_cuda:", use_cuda, "parallel:", parallel)
    train_reader, vocab_size = utils.construct_train_data(
        args.train_dir, args.vocab_path, args.batch_size * get_cards(args))
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    ssr = SequenceSemanticRetrieval(vocab_size, args.embedding_dim,
                                    args.hidden_size)
    # Train program
    train_input_data, cos_pos, avg_cost, acc = ssr.train()

    # Optimization to minimize lost
    optimizer = fluid.optimizer.Adagrad(learning_rate=args.base_lr)
    optimizer.minimize(avg_cost)

    print("run distribute training")
    t = fluid.DistributeTranspiler()
    t.transpile(
        args.trainer_id, pservers=args.endpoints, trainers=args.trainers)
    if args.role == "pserver":
        print("run psever")
        pserver_prog = t.get_pserver_program(args.current_endpoint)
        pserver_startup = t.get_startup_program(args.current_endpoint,
                                                pserver_prog)
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(pserver_startup)
        exe.run(pserver_prog)
    elif args.role == "trainer":
        print("run trainer")
        train_loop(t.get_trainer_program(), avg_cost, acc, train_input_data,
                   place, args, train_reader)


def get_device(args):
    if args.use_cuda:
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES", 1)
        gpu_num = len(gpus.split(','))
        return "gpu", gpu_num
    else:
        threads_num = os.environ.get('NUM_THREADS', 1)
        cpu_num = os.environ.get('CPU_NUM', 1)
        return "cpu", int(cpu_num), int(threads_num)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
