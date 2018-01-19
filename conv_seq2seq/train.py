#coding=utf-8

import os
import sys
import time
import argparse
import distutils.util
import gzip
import numpy as np

import paddle.v2 as paddle
from model import conv_seq2seq
import reader


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle Convolutional Seq2Seq")
    parser.add_argument(
        '--train_data_path',
        type=str,
        required=True,
        help="Path of the training set")
    parser.add_argument(
        '--test_data_path', type=str, help='Path of the test set')
    parser.add_argument(
        '--src_dict_path',
        type=str,
        required=True,
        help='Path of source dictionary')
    parser.add_argument(
        '--trg_dict_path',
        type=str,
        required=True,
        help='Path of target dictionary')
    parser.add_argument(
        '--enc_blocks', type=str, help='Convolution blocks of the encoder')
    parser.add_argument(
        '--dec_blocks', type=str, help='Convolution blocks of the decoder')
    parser.add_argument(
        '--emb_size',
        type=int,
        default=256,
        help='Dimension of word embedding. (default: %(default)s)')
    parser.add_argument(
        '--pos_size',
        type=int,
        default=200,
        help='Total number of the position indexes. (default: %(default)s)')
    parser.add_argument(
        '--drop_rate',
        type=float,
        default=0.,
        help='Dropout rate. (default: %(default)s)')
    parser.add_argument(
        "--use_bn",
        default=False,
        type=distutils.util.strtobool,
        help="Use batch normalization or not. (default: %(default)s)")
    parser.add_argument(
        "--use_gpu",
        default=False,
        type=distutils.util.strtobool,
        help="Use gpu or not. (default: %(default)s)")
    parser.add_argument(
        "--trainer_count",
        default=1,
        type=int,
        help="Trainer number. (default: %(default)s)")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Size of a mini-batch. (default: %(default)s)")
    parser.add_argument(
        '--num_passes',
        type=int,
        default=15,
        help="Number of passes to train. (default: %(default)s)")
    return parser.parse_args()


def create_reader(padding_num,
                  train_data_path,
                  test_data_path=None,
                  src_dict=None,
                  trg_dict=None,
                  pos_size=200,
                  batch_size=32):

    train_reader = paddle.batch(
        reader=paddle.reader.shuffle(
            reader=reader.data_reader(
                data_file=train_data_path,
                src_dict=src_dict,
                trg_dict=trg_dict,
                pos_size=pos_size,
                padding_num=padding_num),
            buf_size=10240),
        batch_size=batch_size)

    test_reader = None
    if test_data_path:
        test_reader = paddle.batch(
            reader=paddle.reader.shuffle(
                reader=reader.data_reader(
                    data_file=test_data_path,
                    src_dict=src_dict,
                    trg_dict=trg_dict,
                    pos_size=pos_size,
                    padding_num=padding_num),
                buf_size=10240),
            batch_size=batch_size)

    return train_reader, test_reader


def train(train_data_path,
          test_data_path,
          src_dict_path,
          trg_dict_path,
          enc_conv_blocks,
          dec_conv_blocks,
          emb_dim=256,
          pos_size=200,
          drop_rate=0.,
          use_bn=False,
          batch_size=32,
          num_passes=15):
    """
    Train the convolution sequence-to-sequence model.    

    :param train_data_path: The path of the training set.
    :type train_data_path: str
    :param test_data_path: The path of the test set.
    :type test_data_path: str
    :param src_dict_path: The path of the source dictionary.
    :type src_dict_path: str
    :param trg_dict_path: The path of the target dictionary.
    :type trg_dict_path: str
    :param enc_conv_blocks: The scale list of the encoder's convolution blocks. And each element of
                            the list contains output dimension and context length of the corresponding
                            convolution block.
    :type enc_conv_blocks: list of tuple
    :param dec_conv_blocks: The scale list of the decoder's convolution blocks. And each element of
                            the list contains output dimension and context length of the corresponding
                            convolution block.
    :type dec_conv_blocks: list of tuple
    :param emb_dim: The dimension of the embedding vector.
    :type emb_dim: int
    :param pos_size: The total number of the position indexes, which means
                     the maximum value of the index is pos_size - 1.
    :type pos_size: int
    :param drop_rate: Dropout rate.
    :type drop_rate: float
    :param use_bn: Whether to use batch normalization or not. False is the default value.
    :type use_bn: bool
    :param batch_size: The size of a mini-batch.
    :type batch_size: int
    :param num_passes: The total number of the passes to train.
    :type num_passes: int
    """
    # load dict
    src_dict = reader.load_dict(src_dict_path)
    trg_dict = reader.load_dict(trg_dict_path)
    src_dict_size = src_dict.__len__()
    trg_dict_size = trg_dict.__len__()

    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, )

    cost = conv_seq2seq(
        src_dict_size=src_dict_size,
        trg_dict_size=trg_dict_size,
        pos_size=pos_size,
        emb_dim=emb_dim,
        enc_conv_blocks=enc_conv_blocks,
        dec_conv_blocks=dec_conv_blocks,
        drop_rate=drop_rate,
        with_bn=use_bn,
        is_infer=False)

    # create parameters and trainer
    parameters = paddle.parameters.create(cost)
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer)

    padding_list = [context_len - 1 for (size, context_len) in dec_conv_blocks]
    padding_num = reduce(lambda x, y: x + y, padding_list)
    train_reader, test_reader = create_reader(
        padding_num=padding_num,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        src_dict=src_dict,
        trg_dict=trg_dict,
        pos_size=pos_size,
        batch_size=batch_size)

    feeding = {
        'src_word': 0,
        'src_word_pos': 1,
        'trg_word': 2,
        'trg_word_pos': 3,
        'trg_next_word': 4
    }

    # create event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 20 == 0:
                cur_time = time.strftime('%Y.%m.%d %H:%M:%S', time.localtime())
                print "[%s]: Pass: %d, Batch: %d, TrainCost: %f, %s" % (
                    cur_time, event.pass_id, event.batch_id, event.cost,
                    event.metrics)
                sys.stdout.flush()

        if isinstance(event, paddle.event.EndPass):
            if test_reader is not None:
                cur_time = time.strftime('%Y.%m.%d %H:%M:%S', time.localtime())
                result = trainer.test(reader=test_reader, feeding=feeding)
                print "[%s]: Pass: %d, TestCost: %f, %s" % (
                    cur_time, event.pass_id, result.cost, result.metrics)
                sys.stdout.flush()
            with gzip.open("output/params.pass-%d.tar.gz" % event.pass_id,
                           'w') as f:
                trainer.save_parameter_to_tar(f)

    if not os.path.exists('output'):
        os.mkdir('output')

    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        num_passes=num_passes,
        feeding=feeding)


def main():
    args = parse_args()
    enc_conv_blocks = eval(args.enc_blocks)
    dec_conv_blocks = eval(args.dec_blocks)

    sys.setrecursionlimit(10000)

    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)

    train(
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        src_dict_path=args.src_dict_path,
        trg_dict_path=args.trg_dict_path,
        enc_conv_blocks=enc_conv_blocks,
        dec_conv_blocks=dec_conv_blocks,
        emb_dim=args.emb_size,
        pos_size=args.pos_size,
        drop_rate=args.drop_rate,
        use_bn=args.use_bn,
        batch_size=args.batch_size,
        num_passes=args.num_passes)


if __name__ == '__main__':
    main()
