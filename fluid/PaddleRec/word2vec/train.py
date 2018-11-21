from __future__ import print_function

import argparse
import logging
import os
import time

import numpy as np

# disable gpu training for this example
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import paddle
import paddle.fluid as fluid

import reader
from network_conf import skip_gram_word2vec

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle CTR example")
    parser.add_argument(
        '--train_data_path',
        type=str,
        default='./data/enwik8',
        help="The path of training dataset")
    parser.add_argument(
        '--dict_path',
        type=str,
        default='./data/enwik8_dict',
        help="The path of data dict")
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='./data/text8',
        help="The path of testing dataset")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help="The size of mini-batch (default:1000)")
    parser.add_argument(
        '--num_passes',
        type=int,
        default=10,
        help="The number of passes to train (default: 10)")
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default='models',
        help='The path for model to store (default: models)')
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=64,
        help='sparse feature hashing space for index processing')

    parser.add_argument(
        '--with_hs',
        action='store_true',
        required=False,
        default=False,
        help='using hierarchical sigmoid, (default: False)')

    parser.add_argument(
        '--with_nce',
        action='store_true',
        required=False,
        default=True,
        help='using negtive sampling, (default: True)')

    parser.add_argument(
        '--max_code_length',
        type=int,
        default=40,
        help='max code length used by hierarchical sigmoid, (default: 40)')
    parser.add_argument(
        '--is_sparse',
        action='store_true',
        required=False,
        default=False,
        help='embedding and nce will use sparse or not, (default: False)')

    parser.add_argument(
        '--is_local',
        type=int,
        default=1,
        help='Local train or distributed train (default: 1)')
    # the following arguments is used for distributed train, if is_local == false, then you should set them
    parser.add_argument(
        '--role',
        type=str,
        default='pserver',  # trainer or pserver
        help='The training role (trainer|pserver) (default: pserver)')
    parser.add_argument(
        '--endpoints',
        type=str,
        default='127.0.0.1:6000',
        help='The pserver endpoints, like: 127.0.0.1:6000,127.0.0.1:6001')
    parser.add_argument(
        '--current_endpoint',
        type=str,
        default='127.0.0.1:6000',
        help='The current pserver endpoint (default: 127.0.0.1:6000)')
    parser.add_argument(
        '--trainer_id',
        type=int,
        default=0,
        help='The current trainer id (default: 0)')
    parser.add_argument(
        '--trainers',
        type=int,
        default=1,
        help='The num of trianers, (default: 1)')

    return parser.parse_args()


def train_loop(args, train_program, reader, py_reader, loss, trainer_id):

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.train((args.with_hs or (not args.with_nce))),
            buf_size=args.batch_size * 100),
        batch_size=args.batch_size)

    py_reader.decorate_paddle_reader(train_reader)

    place = fluid.CPUPlace()

    data_name_list = None

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    start = time.clock()

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = int(os.getenv("NUM_THREADS"))
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

    train_exe = fluid.ParallelExecutor(
        use_cuda=False,
        loss_name=loss.name,
        main_program=train_program,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    profile_state = "CPU"
    profiler_step = 0
    profiler_step_start = 20
    profiler_step_end = 30

    for pass_id in range(args.num_passes):
        epoch_start = time.time()
        py_reader.start()
        batch_id = 0

        try:
            while True:
                if profiler_step == profiler_step_start:
                    fluid.profiler.start_profiler(profile_state)

                loss_val = train_exe.run(fetch_list=[loss.name])
                loss_val = np.mean(loss_val)

                if profiler_step == profiler_step_end:
                    fluid.profiler.stop_profiler('total', 'trainer_profile.log')
                    profiler_step += 1
                else:
                    profiler_step += 1

                if batch_id % 10 == 0:
                    logger.info("TRAIN --> pass: {} batch: {} loss: {}".format(
                        pass_id, batch_id, loss_val.mean() / args.batch_size))
                if batch_id % 1000 == 0 and batch_id != 0:
                    elapsed = (time.clock() - start)
                    logger.info("Time used: {}".format(elapsed))

                if batch_id % 1000 == 0 and batch_id != 0:
                    model_dir = args.model_output_dir + '/batch-' + str(batch_id)
                    if trainer_id == 0:
                        fluid.io.save_inference_model(model_dir, data_name_list, [loss], exe)
                batch_id += 1

        except fluid.core.EOFException:
            py_reader.reset()
            epoch_end = time.time()
            print("Epoch: {0}, Train total expend: {1} ".format(pass_id, epoch_end - epoch_start))

            model_dir = args.model_output_dir + '/pass-' + str(pass_id)
            if trainer_id == 0:
                fluid.io.save_inference_model(model_dir, data_name_list, [loss], exe)


def train():
    args = parse_args()

    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

    word2vec_reader = reader.Word2VecReader(args.dict_path,
                                            args.train_data_path)

    logger.info("dict_size: {}".format(word2vec_reader.dict_size))

    loss, py_reader = skip_gram_word2vec(
        word2vec_reader.dict_size, word2vec_reader.word_frequencys,
        args.embedding_size, args.max_code_length,
        args.with_hs, args.with_nce, is_sparse=args.is_sparse)

    optimizer = fluid.optimizer.Adam(learning_rate=1e-3)
    optimizer.minimize(loss)

    if args.is_local:
        logger.info("run local training")
        main_program = fluid.default_main_program()
        train_loop(args, main_program, word2vec_reader, py_reader, loss, 0)
    else:
        logger.info("run dist training")
        t = fluid.DistributeTranspiler()
        t.transpile(
            args.trainer_id, pservers=args.endpoints, trainers=args.trainers)
        if args.role == "pserver":
            logger.info("run pserver")
            prog = t.get_pserver_program(args.current_endpoint)
            startup = t.get_startup_program(
                args.current_endpoint, pserver_program=prog)
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup)
            exe.run(prog)
        elif args.role == "trainer":
            logger.info("run trainer")
            train_prog = t.get_trainer_program()
            train_loop(args, train_prog, word2vec_reader, py_reader, loss, args.trainer_id)


if __name__ == '__main__':
    train()
