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
from paddle.fluid.executor import global_scope

import reader
from network_conf import skip_gram_word2vec
from infer import inference_test

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle Word2vec example")
    parser.add_argument(
        '--train_data_path',
        type=str,
        default='./data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled',
        help="The path of training dataset")
    parser.add_argument(
        '--dict_path',
        type=str,
        default='./data/1-billion_dict',
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
        help="The size of mini-batch (default:100)")
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
        default=False,
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
        '--with_Adam',
        action='store_true',
        required=False,
        default=False,
        help='Using Adam as optimizer or not, (default: False)')

    parser.add_argument(
        '--is_local',
        action='store_true',
        required=False,
        default=False,
        help='Local train or not, (default: False)')

    parser.add_argument(
        '--with_speed',
        action='store_true',
        required=False,
        default=False,
        help='print speed or not , (default: False)')

    parser.add_argument(
        '--with_infer_test',
        action='store_true',
        required=False,
        default=False,
        help='Do inference every 100 batches , (default: False)')

    parser.add_argument(
        '--rank_num',
        type=int,
        default=4,
        help="find rank_num-nearest result for test (default: 4)")

    return parser.parse_args()


def train_loop(args, train_program, reader, py_reader, loss, trainer_id):
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.train((args.with_hs or (not args.with_nce))),
            buf_size=args.batch_size * 100),
        batch_size=args.batch_size)

    py_reader.decorate_paddle_reader(train_reader)

    place = fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    exec_strategy = fluid.ExecutionStrategy()

    print("CPU_NUM:" + str(os.getenv("CPU_NUM")))
    exec_strategy.num_threads = int(os.getenv("CPU_NUM"))

    build_strategy = fluid.BuildStrategy()
    if int(os.getenv("CPU_NUM")) > 1:
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
        start = time.clock()

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

                if batch_id % 50 == 0:
                    logger.info(
                        "TRAIN --> pass: {} batch: {} loss: {} reader queue:{}".
                        format(pass_id, batch_id,
                               loss_val.mean() / args.batch_size,
                               py_reader.queue.size()))
                if args.with_speed:
                    if batch_id % 1000 == 0 and batch_id != 0:
                        elapsed = (time.clock() - start)
                        start = time.clock()
                        samples = 1001 * args.batch_size * int(
                            os.getenv("CPU_NUM"))
                        logger.info("Time used: {}, Samples/Sec: {}".format(
                            elapsed, samples / elapsed))
                # calculate infer result each 100 batches when using --with_infer_test
                if args.with_infer_test:
                    if batch_id % 1000 == 0 and batch_id != 0:
                        model_dir = args.model_output_dir + '/batch-' + str(
                            batch_id)
                        inference_test(global_scope(), model_dir, args)

                if batch_id % 500000 == 0 and batch_id != 0:
                    model_dir = args.model_output_dir + '/batch-' + str(
                        batch_id)
                    fluid.io.save_persistables(executor=exe, dirname=model_dir)
                    with open(model_dir + "/_success", 'w+') as f:
                        f.write(str(batch_id))
                batch_id += 1

        except fluid.core.EOFException:
            py_reader.reset()
            epoch_end = time.time()
            logger.info("Epoch: {0}, Train total expend: {1} ".format(
                pass_id, epoch_end - epoch_start))

            model_dir = args.model_output_dir + '/pass-' + str(pass_id)
            if trainer_id == 0:
                fluid.io.save_persistables(executor=exe, dirname=model_dir)
                with open(model_dir + "/_success", 'w+') as f:
                    f.write(str(pass_id))


def GetFileList(data_path):
    return os.listdir(data_path)


def train(args):

    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

    filelist = GetFileList(args.train_data_path)
    word2vec_reader = None
    if args.is_local or os.getenv("PADDLE_IS_LOCAL", "1") == "1":
        word2vec_reader = reader.Word2VecReader(
            args.dict_path, args.train_data_path, filelist, 0, 1)
    else:
        trainer_id = int(os.environ["PADDLE_TRAINER_ID"])
        trainers = int(os.environ["PADDLE_TRAINERS"])
        word2vec_reader = reader.Word2VecReader(args.dict_path,
                                                args.train_data_path, filelist,
                                                trainer_id, trainer_num)

    logger.info("dict_size: {}".format(word2vec_reader.dict_size))
    loss, py_reader = skip_gram_word2vec(
        word2vec_reader.dict_size,
        word2vec_reader.word_frequencys,
        args.embedding_size,
        args.max_code_length,
        args.with_hs,
        args.with_nce,
        is_sparse=args.is_sparse)

    optimizer = None
    if args.with_Adam:
        optimizer = fluid.optimizer.Adam(learning_rate=1e-4)
    else:
        optimizer = fluid.optimizer.SGD(learning_rate=1e-4)

    optimizer.minimize(loss)

    # do local training 
    if args.is_local or os.getenv("PADDLE_IS_LOCAL", "1") == "1":
        logger.info("run local training")
        main_program = fluid.default_main_program()

        with open("local.main.proto", "w") as f:
            f.write(str(main_program))

        train_loop(args, main_program, word2vec_reader, py_reader, loss, 0)
    # do distribute training
    else:
        logger.info("run dist training")

        trainer_id = int(os.environ["PADDLE_TRAINER_ID"])
        trainers = int(os.environ["PADDLE_TRAINERS"])
        training_role = os.environ["PADDLE_TRAINING_ROLE"]

        port = os.getenv("PADDLE_PSERVER_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVER_IPS", "")
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)
        current_endpoint = os.getenv("PADDLE_CURRENT_IP", "") + ":" + port

        config = fluid.DistributeTranspilerConfig()
        config.slice_var_up = False
        t = fluid.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id,
            pservers=pserver_endpoints,
            trainers=trainers,
            sync_mode=True)

        if training_role == "PSERVER":
            logger.info("run pserver")
            prog = t.get_pserver_program(current_endpoint)
            startup = t.get_startup_program(
                current_endpoint, pserver_program=prog)

            with open("pserver.main.proto.{}".format(os.getenv("CUR_PORT")),
                      "w") as f:
                f.write(str(prog))

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup)
            exe.run(prog)
        elif training_role == "TRAINER":
            logger.info("run trainer")
            train_prog = t.get_trainer_program()

            with open("trainer.main.proto.{}".format(trainer_id), "w") as f:
                f.write(str(train_prog))

            train_loop(args, train_prog, word2vec_reader, py_reader, loss,
                       trainer_id)


def env_declar():
    print("********  Rename Cluster Env to PaddleFluid Env ********")

    print("Content-Type: text/plain\n\n")
    for key in os.environ.keys():
        print("%30s %s \n" % (key, os.environ[key]))

    if os.environ["TRAINING_ROLE"] == "PSERVER" or os.environ[
            "PADDLE_IS_LOCAL"] == "0":
        os.environ["PADDLE_TRAINING_ROLE"] = os.environ["TRAINING_ROLE"]
        os.environ["PADDLE_PSERVER_PORT"] = os.environ["PADDLE_PORT"]
        os.environ["PADDLE_PSERVER_IPS"] = os.environ["PADDLE_PSERVERS"]
        os.environ["PADDLE_TRAINERS"] = os.environ["PADDLE_TRAINERS_NUM"]
        os.environ["PADDLE_CURRENT_IP"] = os.environ["POD_IP"]
        os.environ["PADDLE_TRAINER_ID"] = os.environ["PADDLE_TRAINER_ID"]
        # we set the thread number same as CPU number
        os.environ["CPU_NUM"] = "12"

    print("Content-Type: text/plain\n\n")
    for key in os.environ.keys():
        print("%30s %s \n" % (key, os.environ[key]))

    print("******  Rename Cluster Env to PaddleFluid Env END ******")


if __name__ == '__main__':
    args = parse_args()
    if args.is_local:
        pass
    else:
        env_declar()
    train(args)
