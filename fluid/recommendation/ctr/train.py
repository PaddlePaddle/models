import os
import logging
import argparse

import paddle.fluid as fluid

from network_conf import DeepFM
import reader
import paddle

logging.basicConfig()
logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle DeepFM example")
    parser.add_argument(
        '--train_data_path',
        type=str,
        default='./data/train.txt',
        help="The path of training dataset")
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='./data/test.txt',
        help="The path of testing dataset")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help="The size of mini-batch (default:1000)")
    parser.add_argument(
        '--factor_size',
        type=int,
        default=10,
        help="The factor size for the factorization machine (default:10)")
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

    return parser.parse_args()


def train():
    args = parse_args()

    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

    loss, data_list, auc_var, batch_auc_var = DeepFM(args.factor_size)
    optimizer = fluid.optimizer.Adam(learning_rate=1e-4)
    optimize_ops, params_grads = optimizer.minimize(loss)

    dataset = reader.Dataset()
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            dataset.train(args.train_data_path),
            buf_size=args.batch_size * 100),
        batch_size=args.batch_size)
    place = fluid.CPUPlace()

    feeder = fluid.DataFeeder(feed_list=data_list, place=place)
    data_name_list = [var.name for var in data_list]

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    for pass_id in range(args.num_passes):
        batch_id = 0
        for data in train_reader():
            loss_val, auc_val, batch_auc_val = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[loss, auc_var, batch_auc_var]
            )
            print('pass:' + str(pass_id) + ' batch:' + str(batch_id) + ' loss: ' + str(loss_val) + " auc: " + str(auc_val) + " batch_auc: " + str(batch_auc_val))
            batch_id += 1
            if batch_id % 100 == 0 and batch_id != 0:
                model_dir = 'output/batch-' + str(batch_id)
                fluid.io.save_inference_model(model_dir, data_name_list, [loss, auc_var], exe)
        model_dir = 'output/pass-' + str(pass_id)
        fluid.io.save_inference_model(model_dir, data_name_list, [loss_var, auc_var], exe)


if __name__ == '__main__':
    train()
