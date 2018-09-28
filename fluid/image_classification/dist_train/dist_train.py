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
from args import *
from reader import train, val

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
            model_def = models.__dict__[args.model]()
            predict = model_def.net(input, class_dim=class_dim)

            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(x=cost)

            batch_acc1 = fluid.layers.accuracy(input=predict, label=label, k=1)
            batch_acc5 = fluid.layers.accuracy(input=predict, label=label, k=5)

            # configure optimize
            optimizer = None
            if is_train:

                total_images = 1281167 / trainer_count

                step = int(total_images / (args.batch_size * args.gpus) + 1)
                epochs = [30, 60, 90]
                bd = [step * e for e in epochs]
                base_lr = args.learning_rate
                lr = []
                lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
                optimizer = fluid.optimizer.Momentum(
                    learning_rate=fluid.layers.piecewise_decay(
                        boundaries=bd, values=lr),
                    momentum=0.9,
                    regularization=fluid.regularizer.L2Decay(1e-4))
                optimizer.minimize(avg_cost)

                if args.memory_optimize:
                    fluid.memory_optimize(main_prog)

    batched_reader = None
    pyreader.decorate_paddle_reader(
        paddle.batch(
            reader if args.no_random else paddle.reader.shuffle(
                reader, buf_size=5120),
            batch_size=args.batch_size))

    return avg_cost, optimizer, [batch_acc1,
                                 batch_acc5], batched_reader, pyreader

def append_nccl2_prepare(trainer_id, startup_prog):
    if trainer_id >= 0:
        # append gen_nccl_id at the end of startup program
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        port = os.getenv("PADDLE_PSERVER_PORT")
        worker_ips = os.getenv("PADDLE_TRAINER_IPS")
        worker_endpoints = []
        for ip in worker_ips.split(","):
            worker_endpoints.append(':'.join([ip, port]))
        num_trainers = len(worker_endpoints)
        current_endpoint = os.getenv("PADDLE_CURRENT_IP") + ":" + port
        worker_endpoints.remove(current_endpoint)

        nccl_id_var = startup_prog.global_block().create_var(
            name="NCCLID",
            persistable=True,
            type=fluid.core.VarDesc.VarType.RAW)
        startup_prog.global_block().append_op(
            type="gen_nccl_id",
            inputs={},
            outputs={"NCCLID": nccl_id_var},
            attrs={
                "endpoint": current_endpoint,
                "endpoint_list": worker_endpoints,
                "trainer_id": trainer_id
            })
        return nccl_id_var, num_trainers, trainer_id
    else:
        raise Exception("must set positive PADDLE_TRAINER_ID env variables for "
                        "nccl-based dist train.")


def dist_transpile(trainer_id, args, train_prog, startup_prog):
    if trainer_id < 0:
        return None, None

    # the port of all pservers, needed by both trainer and pserver
    port = os.getenv("PADDLE_PSERVER_PORT", "6174")
    # comma separated ips of all pservers, needed by trainer and
    # pserver
    pserver_ips = os.getenv("PADDLE_PSERVER_IPS", "")
    eplist = []
    for ip in pserver_ips.split(","):
        eplist.append(':'.join([ip, port]))
    pserver_endpoints = ",".join(eplist)
    # total number of workers/trainers in the job, needed by
    # trainer and pserver
    trainers = int(os.getenv("PADDLE_TRAINERS"))
    # the IP of the local machine, needed by pserver only
    current_endpoint = os.getenv("PADDLE_CURRENT_IP", "") + ":" + port
    # the role, should be either PSERVER or TRAINER
    training_role = os.getenv("PADDLE_TRAINING_ROLE")

    config = fluid.DistributeTranspilerConfig()
    config.slice_var_up = not args.no_split_var
    t = fluid.DistributeTranspiler(config=config)
    t.transpile(
        trainer_id,
        # NOTE: *MUST* use train_prog, for we are using with guard to
        # generate different program for train and test.
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


def test_parallel(exe, test_args, args, test_prog, feeder):
    acc_evaluators = []
    for i in six.moves.xrange(len(test_args[2])):
        acc_evaluators.append(fluid.metrics.Accuracy())

    to_fetch = [v.name for v in test_args[2]]
    test_args[4].start()
    while True:
        try:
            acc_rets = exe.run(fetch_list=to_fetch)
            for i, e in enumerate(acc_evaluators):
                e.update(
                    value=np.array(acc_rets[i]), weight=args.batch_size)
        except fluid.core.EOFException as eof:
            test_args[4].reset()
            break

    return [e.eval() for e in acc_evaluators]


# NOTE: only need to benchmark using parallelexe
def train_parallel(train_args, test_args, args, train_prog, test_prog,
                   startup_prog, nccl_id_var, num_trainers, trainer_id):
    over_all_start = time.time()
    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    feeder = None

    if nccl_id_var and trainer_id == 0:
        #FIXME(wuyi): wait other trainer to start listening
        time.sleep(30)

    startup_exe = fluid.Executor(place)
    startup_exe.run(startup_prog)
    strategy = fluid.ExecutionStrategy()
    strategy.num_threads = args.cpus
    strategy.allow_op_delay = False
    build_strategy = fluid.BuildStrategy()
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

    exe = fluid.ParallelExecutor(
        True,
        avg_loss.name,
        main_program=train_prog,
        exec_strategy=strategy,
        build_strategy=build_strategy,
        num_trainers=num_trainers,
        trainer_id=trainer_id)

    if not args.no_test:
        if args.update_method == "pserver":
            test_scope = None
        else:
            # NOTE: use an empty scope to avoid test exe using NCCLID
            test_scope = fluid.Scope()
        test_exe = fluid.ParallelExecutor(
            True, main_program=test_prog, share_vars_from=exe)

    pyreader = train_args[4]
    for pass_id in range(args.pass_num):
        num_samples = 0
        iters = 0
        start_time = time.time()
        batch_id = 0
        pyreader.start()
        while True:
            if iters == args.iterations:
                break

            if iters == args.skip_batch_num:
                start_time = time.time()
                num_samples = 0
            fetch_list = [avg_loss.name]
            acc_name_list = [v.name for v in train_args[2]]
            fetch_list.extend(acc_name_list)

            try:
                fetch_ret = exe.run(fetch_list)
            except fluid.core.EOFException as eof:
                break
            except fluid.core.EnforceNotMet as ex:
                traceback.print_exc()
                break
            num_samples += args.batch_size * args.gpus

            iters += 1
            if batch_id % 1 == 0:
                fetched_data = [np.mean(np.array(d)) for d in fetch_ret]
                print("Pass %d, batch %d, loss %s, accucacys: %s" %
                      (pass_id, batch_id, fetched_data[0], fetched_data[1:]))
            batch_id += 1

        print_train_time(start_time, time.time(), num_samples)
        pyreader.reset() # reset reader handle

        if not args.no_test and test_args[2]:
            test_feeder = None
            test_ret = test_parallel(test_exe, test_args, args, test_prog,
                                     test_feeder)
            print("Pass: %d, Test Accuracy: %s\n" %
                  (pass_id, [np.mean(np.array(v)) for v in test_ret]))

    startup_exe.close()
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
    if args.no_random:
        fluid.default_startup_program().random_seed = 1

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
        if args.gpus > 1 and os.getenv("PADDLE_TRAINING_ROLE") == "TRAINER":
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
