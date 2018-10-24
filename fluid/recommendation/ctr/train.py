from __future__ import print_function
import argparse
import os
import time

import paddle
import paddle.fluid as fluid

import reader
from network_conf import ctr_dnn_model


def print_log(log_str):
    time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(str(time_stamp) + " " + log_str)


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle CTR example")
    parser.add_argument(
        '--train_data_path',
        type=str,
        default='./data/train.txt',
        help="The path of training dataset")
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='./data/valid.txt',
        help="The path of testing dataset")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help="The size of mini-batch (default:1000)")
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=10,
        help="The size for embedding layer (default:10)")
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
        '--is_local',
        type=int,
        default=1,
        help='Local train or distributed train (default: 1)')
    # the following arguments is used for distributed train, if is_local == false, then you should set them
    parser.add_argument(
        '--role',
        type=str,
        default='pserver', # trainer or pserver
        help='The path for model to store (default: models)')
    parser.add_argument(
        '--endpoints',
        type=str,
        default='127.0.0.1:6000',
        help='The pserver endpoints, like: 127.0.0.1:6000,127.0.0.1:6001')
    parser.add_argument(
        '--current_endpoint',
        type=str,
        default='127.0.0.1:6000',
        help='The path for model to store (default: 127.0.0.1:6000)')
    parser.add_argument(
        '--trainer_id',
        type=int,
        default=0,
        help='The path for model to store (default: models)')
    parser.add_argument(
        '--trainers',
        type=int,
        default=1,
        help='The num of trianers, (default: 1)')

    return parser.parse_args()


def train_loop(args, train_program, data_list, loss, auc_var, batch_auc_var):
    dataset = reader.Dataset()
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            dataset.train([args.train_data_path]),
            buf_size=args.batch_size * 100),
        batch_size=args.batch_size)
    place = fluid.CPUPlace()

    feeder = fluid.DataFeeder(feed_list=data_list, place=place)
    data_name_list = [var.name for var in data_list]

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    for pass_id in range(args.num_passes):
        for batch_id, data in enumerate(train_reader()):
            loss_val, auc_val, batch_auc_val = exe.run(
                train_program,
                feed=feeder.feed(data),
                fetch_list=[loss, auc_var, batch_auc_var]
            )
            print_log("TRAIN --> pass: {} batch: {} loss: {} auc: {}, batch_auc: {}"
                      .format(pass_id, batch_id, loss_val, auc_val, batch_auc_val))
            if batch_id % 1000 == 0 and batch_id != 0:
                model_dir = args.model_output_dir + '/batch-' + str(batch_id)
                fluid.io.save_inference_model(model_dir, data_name_list, [loss, auc_var], exe)
        model_dir = args.model_output_dir + '/pass-' + str(pass_id)
        fluid.io.save_inference_model(model_dir, data_name_list, [loss, auc_var], exe)


def train():
    args = parse_args()

    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

    loss, data_list, auc_var, batch_auc_var = ctr_dnn_model(args.embedding_size)
    optimizer = fluid.optimizer.Adam(learning_rate=1e-4)
    optimizer.minimize(loss)

    if args.is_local:
        print_log("run local training")
        main_program = fluid.default_main_program()
        train_loop(args, main_program, data_list, loss, auc_var, batch_auc_var)
    else:
        print_log("run dist training")
        t = fluid.DistributeTranspiler()
        t.transpile(args.trainer_id, pservers=args.endpoints, trainers=args.trainers)
        if args.role == "pserver":
            print_log("run pserver")
            prog = t.get_pserver_program(args.current_endpoint)
            startup = t.get_startup_program(args.current_endpoint, pserver_program=prog)
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup)
            exe.run(prog)
        elif args.role == "trainer":
            print_log("run trainer")
            train_prog = t.get_trainer_program()
            train_loop(args, train_prog, data_list, loss, auc_var, batch_auc_var)


if __name__ == '__main__':
    train()
