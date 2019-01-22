# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import time
import os
import traceback

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import six
import sys
sys.path.append("..")
import models
from reader import train, val

def parse_args():
    parser = argparse.ArgumentParser('Distributed Image Classification Training.')
    parser.add_argument(
        '--model',
        type=str,
        default='DistResNet',
        help='The model to run.')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='The minibatch size per device.')
    parser.add_argument(
        '--multi_batch_repeat', type=int, default=1, help='Batch merge repeats.')
    parser.add_argument(
        '--learning_rate', type=float, default=0.1, help='The learning rate.')
    parser.add_argument(
        '--pass_num', type=int, default=90, help='The number of passes.')
    parser.add_argument(
        '--data_format',
        type=str,
        default='NCHW',
        choices=['NCHW', 'NHWC'],
        help='The data data_format, now only support NCHW.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='If gpus > 1, will use ParallelExecutor to run, else use Executor.')
    parser.add_argument(
        '--cpus',
        type=int,
        default=1,
        help='If cpus > 1, will set ParallelExecutor to use multiple threads.')
    parser.add_argument(
        '--no_test',
        action='store_true',
        help='If set, do not test the testset during training.')
    parser.add_argument(
        '--memory_optimize',
        action='store_true',
        help='If set, optimize runtime memory before start.')
    parser.add_argument(
        '--update_method',
        type=str,
        default='local',
        choices=['local', 'pserver', 'nccl2'],
        help='Choose parameter update method, can be local, pserver, nccl2.')
    parser.add_argument(
        '--no_split_var',
        action='store_true',
        default=False,
        help='Whether split variables into blocks when update_method is pserver')
    parser.add_argument(
        '--async_mode',
        action='store_true',
        default=False,
        help='Whether start pserver in async mode to support ASGD')
    parser.add_argument(
        '--reduce_strategy',
        type=str,
        choices=['reduce', 'all_reduce'],
        default='all_reduce',
        help='Specify the reduce strategy, can be reduce, all_reduce')
    parser.add_argument(
        '--data_dir',
        type=str,
        default="../data/ILSVRC2012",
        help="The ImageNet dataset root dir."
    )
    args = parser.parse_args()
    return args

def get_model(args, is_train, main_prog, startup_prog):
    pyreader = None
    class_dim = 1000
    if args.data_format == 'NCHW':
        dshape = [3, 224, 224]
    else:
        dshape = [224, 224, 3]
    if is_train:
        reader = train(data_dir=args.data_dir)
    else:
        reader = val(data_dir=args.data_dir)

    trainer_count = int(os.getenv("PADDLE_TRAINERS", "1"))
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            pyreader = fluid.layers.py_reader(
                capacity=args.batch_size * args.gpus,
                shapes=([-1] + dshape, (-1, 1)),
                dtypes=('float32', 'int64'),
                name="train_reader" if is_train else "test_reader",
                use_double_buffer=True)
            input, label = fluid.layers.read_file(pyreader)
            model_def = models.__dict__[args.model](layers=50, is_train=is_train)
            predict = model_def.net(input, class_dim=class_dim)

            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(x=cost)

            batch_acc1 = fluid.layers.accuracy(input=predict, label=label, k=1)
            batch_acc5 = fluid.layers.accuracy(input=predict, label=label, k=5)

            optimizer = None
            if is_train:
                start_lr = args.learning_rate
                # n * worker * repeat
                end_lr = args.learning_rate * trainer_count * args.multi_batch_repeat
                total_images = 1281167 / trainer_count
                step = int(total_images / (args.batch_size * args.gpus * args.multi_batch_repeat) + 1)
                warmup_steps = step * 5  # warmup 5 passes
                epochs = [30, 60, 80]
                bd = [step * e for e in epochs]
                base_lr = end_lr
                lr = []
                lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]

                optimizer = fluid.optimizer.Momentum(
                    learning_rate=models.learning_rate.lr_warmup(
                        fluid.layers.piecewise_decay(
                            boundaries=bd, values=lr),
                        warmup_steps, start_lr, end_lr),
                    momentum=0.9,
                    regularization=fluid.regularizer.L2Decay(1e-4))
                optimizer.minimize(avg_cost)

    batched_reader = None
    pyreader.decorate_paddle_reader(
        paddle.batch(
            reader,
            batch_size=args.batch_size))

    return avg_cost, optimizer, [batch_acc1,
                                 batch_acc5], batched_reader, pyreader

def append_nccl2_prepare(trainer_id, startup_prog):
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
    port = os.getenv("PADDLE_PSERVER_PORT")
    worker_ips = os.getenv("PADDLE_TRAINER_IPS")
    worker_endpoints = []
    for ip in worker_ips.split(","):
        worker_endpoints.append(':'.join([ip, port]))
    current_endpoint = os.getenv("PADDLE_CURRENT_IP") + ":" + port

    config = fluid.DistributeTranspilerConfig()
    config.mode = "nccl2"
    t = fluid.DistributeTranspiler(config=config)
    t.transpile(trainer_id, trainers=','.join(worker_endpoints),
        current_endpoint=current_endpoint,
        startup_program=startup_prog)


def dist_transpile(trainer_id, args, train_prog, startup_prog):
    port = os.getenv("PADDLE_PSERVER_PORT", "6174")
    pserver_ips = os.getenv("PADDLE_PSERVER_IPS", "")
    eplist = []
    for ip in pserver_ips.split(","):
        eplist.append(':'.join([ip, port]))
    pserver_endpoints = ",".join(eplist)
    trainers = int(os.getenv("PADDLE_TRAINERS"))
    current_endpoint = os.getenv("PADDLE_CURRENT_IP", "") + ":" + port
    training_role = os.getenv("PADDLE_TRAINING_ROLE")

    config = fluid.DistributeTranspilerConfig()
    config.slice_var_up = not args.no_split_var
    t = fluid.DistributeTranspiler(config=config)
    t.transpile(
        trainer_id,
        program=train_prog,
        pservers=pserver_endpoints,
        trainers=trainers,
        sync_mode=not args.async_mode,
        startup_program=startup_prog)
    if training_role == "PSERVER":
        pserver_program = t.get_pserver_program(current_endpoint)
        pserver_startup_program = t.get_startup_program(
            current_endpoint, pserver_program, startup_program=startup_prog)
        return pserver_program, pserver_startup_program
    elif training_role == "TRAINER":
        train_program = t.get_trainer_program()
        return train_program, startup_prog
    else:
        raise ValueError(
            'PADDLE_TRAINING_ROLE environment variable must be either TRAINER or PSERVER'
        )

def append_bn_repeat_init_op(main_prog, startup_prog, num_repeats):
    repeat_vars = set()
    for op in main_prog.global_block().ops:
        if op.type == "batch_norm":
            repeat_vars.add(op.input("Mean")[0])
            repeat_vars.add(op.input("Variance")[0])
    
    for i in range(num_repeats):
        for op in startup_prog.global_block().ops:
            if op.type == "fill_constant":
                for oname in op.output_arg_names:
                    if oname in repeat_vars:
                        var = startup_prog.global_block().var(oname)
                        repeat_var_name = "%s.repeat.%d" % (oname, i)
                        repeat_var = startup_prog.global_block().create_var(
                            name=repeat_var_name,
                            type=var.type,
                            dtype=var.dtype,
                            shape=var.shape,
                            persistable=var.persistable
                        )
                        main_prog.global_block()._clone_variable(repeat_var)
                        startup_prog.global_block().append_op(
                            type="fill_constant",
                            inputs={},
                            outputs={"Out": repeat_var},
                            attrs=op.all_attrs()
                        )


def copyback_repeat_bn_params(main_prog):
    repeat_vars = set()
    for op in main_prog.global_block().ops:
        if op.type == "batch_norm":
            repeat_vars.add(op.input("Mean")[0])
            repeat_vars.add(op.input("Variance")[0])
    for vname in repeat_vars:
        real_var = fluid.global_scope().find_var("%s.repeat.0" % vname).get_tensor()
        orig_var = fluid.global_scope().find_var(vname).get_tensor()
        orig_var.set(np.array(real_var), fluid.CUDAPlace(0)) # test on GPU0


def test_single(exe, test_args, args, test_prog):
    acc_evaluators = []
    for i in xrange(len(test_args[2])):
        acc_evaluators.append(fluid.metrics.Accuracy())

    to_fetch = [v.name for v in test_args[2]]
    test_args[4].start()
    while True:
        try:
            acc_rets = exe.run(program=test_prog, fetch_list=to_fetch)
            for i, e in enumerate(acc_evaluators):
                e.update(
                    value=np.array(acc_rets[i]), weight=args.batch_size)
        except fluid.core.EOFException as eof:
            test_args[4].reset()
            break

    return [e.eval() for e in acc_evaluators]


def train_parallel(train_args, test_args, args, train_prog, test_prog,
                   startup_prog, nccl_id_var, num_trainers, trainer_id):
    over_all_start = time.time()
    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)

    if nccl_id_var and trainer_id == 0:
        #FIXME(wuyi): wait other trainer to start listening
        time.sleep(30)

    exe = fluid.Executor(place)
    if args.multi_batch_repeat > 1:
        append_bn_repeat_init_op(train_prog, startup_prog, args.multi_batch_repeat)
    exe.run(startup_prog)
    strategy = fluid.ExecutionStrategy()
    strategy.num_threads = args.cpus
    strategy.allow_op_delay = False
    build_strategy = fluid.BuildStrategy()
    if args.multi_batch_repeat > 1:
        pass_builder = build_strategy._create_passes_from_strategy()
        mypass = pass_builder.insert_pass(
            len(pass_builder.all_passes()) - 2, "multi_batch_merge_pass")
        mypass.set_int("num_repeats", args.multi_batch_repeat)
    if args.reduce_strategy == "reduce":
        build_strategy.reduce_strategy = fluid.BuildStrategy(
        ).ReduceStrategy.Reduce
    else:
        build_strategy.reduce_strategy = fluid.BuildStrategy(
        ).ReduceStrategy.AllReduce

    avg_loss = train_args[0]

    if args.update_method == "pserver":
        # parameter server mode distributed training, merge
        # gradients on local server, do not initialize
        # ParallelExecutor with multi server all-reduce mode.
        num_trainers = 1
        trainer_id = 0

    build_strategy.num_trainers = num_trainers
    build_strategy.trainer_id = trainer_id
    train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name=avg_loss.name,
        exec_strategy=strategy,
        build_strategy=build_strategy)

    pyreader = train_args[4]
    for pass_id in range(args.pass_num):
        num_samples = 0
        start_time = time.time()
        batch_id = 0
        pyreader.start()
        while True:
            fetch_list = [avg_loss.name]
            acc_name_list = [v.name for v in train_args[2]]
            fetch_list.extend(acc_name_list)
            try:
                if batch_id % 30 == 0:
                    fetch_ret = exe.run(train_prog, fetch_list=fetch_list)
                else:
                    fetch_ret = exe.run(train_prog, fetch_list=[])
            except fluid.core.EOFException as eof:
                break
            except fluid.core.EnforceNotMet as ex:
                traceback.print_exc()
                break
            num_samples += args.batch_size * args.gpus

            if batch_id % 30 == 0:
                fetched_data = [np.mean(np.array(d)) for d in fetch_ret]
                print("Pass %d, batch %d, loss %s, accucacys: %s" %
                      (pass_id, batch_id, fetched_data[0], fetched_data[1:]))
            batch_id += 1

        print_train_time(start_time, time.time(), num_samples)
        pyreader.reset()

        if not args.no_test and test_args[2]:
            if args.multi_batch_repeat > 1:
                copyback_repeat_bn_params(train_prog)
            test_ret = test_single(exe, test_args, args, test_prog)
            print("Pass: %d, Test Accuracy: %s\n" %
                  (pass_id, [np.mean(np.array(v)) for v in test_ret]))

    exe.close()
    print("total train time: ", time.time() - over_all_start)


def print_arguments(args):
    print('----------- Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def print_train_time(start_time, end_time, num_samples):
    train_elapsed = end_time - start_time
    examples_per_sec = num_samples / train_elapsed
    print('\nTotal examples: %d, total time: %.5f, %.5f examples/sed\n' %
          (num_samples, train_elapsed, examples_per_sec))


def print_paddle_envs():
    print('----------- Configuration envs -----------')
    for k in os.environ:
        if "PADDLE_" in k:
            print("ENV %s:%s" % (k, os.environ[k]))
    print('------------------------------------------------')


def main():
    args = parse_args()
    print_arguments(args)
    print_paddle_envs()

    # the unique trainer id, starting from 0, needed by trainer
    # only
    nccl_id_var, num_trainers, trainer_id = (
        None, 1, int(os.getenv("PADDLE_TRAINER_ID", "0")))

    train_prog = fluid.Program()
    test_prog = fluid.Program()
    startup_prog = fluid.Program()

    train_args = list(get_model(args, True, train_prog, startup_prog))
    test_args = list(get_model(args, False, test_prog, startup_prog))

    all_args = [train_args, test_args, args]

    if args.update_method == "pserver":
        train_prog, startup_prog = dist_transpile(trainer_id, args, train_prog,
                                                  startup_prog)
        if not train_prog:
            raise Exception(
                "Must configure correct environments to run dist train.")
        all_args.extend([train_prog, test_prog, startup_prog])
        if os.getenv("PADDLE_TRAINING_ROLE") == "TRAINER":
            all_args.extend([nccl_id_var, num_trainers, trainer_id])
            train_parallel(*all_args)
        elif os.getenv("PADDLE_TRAINING_ROLE") == "PSERVER":
            # start pserver with Executor
            server_exe = fluid.Executor(fluid.CPUPlace())
            server_exe.run(startup_prog)
            server_exe.run(train_prog)
        exit(0)

    # for other update methods, use default programs
    all_args.extend([train_prog, test_prog, startup_prog])

    if args.update_method == "nccl2":
        nccl_id_var, num_trainers, trainer_id = append_nccl2_prepare(
            trainer_id, startup_prog)

    all_args.extend([nccl_id_var, num_trainers, trainer_id])
    train_parallel(*all_args)

if __name__ == "__main__":
    main()
