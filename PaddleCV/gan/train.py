from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from util import config, utility
from data_reader import reader_creator, mnist_reader_creator
import os
import sys
import six
import time
import numpy as np
import paddle
import paddle.fluid as fluid


def train(cfg):
    shuffle = cfg.shuffle
    if cfg.run_ce:
        np.random.seed(10)
        fluid.default_startup_program().random_seed = 73
        shuffle = False

    if cfg.dataset == 'mnist':
        train_images = os.path.join(cfg.data_dir, cfg.dataset,
                                    "train-images-idx3-ubyte.gz")
        train_labels = os.path.join(cfg.data_dir, cfg.dataset,
                                    "train-labels-idx1-ubyte.gz")

        if cfg.run_ce:
            train_reader = paddle.batch(
                mnist_reader_creator(train_images, train_labels, 100),
                batch_size=cfg.batch_size)
        else:
            train_reader = paddle.batch(
                paddle.reader.shuffle(
                    mnist_reader_creator(train_images, train_labels, 100),
                    buf_size=60000),
                batch_size=cfg.batch_size)
    else:
        if cfg.model_net == 'cyclegan':
            dataset_dir = os.path.join(cfg.data_dir, cfg.dataset)
            trainA_list = os.path.join(dataset_dir, "trainA.txt")
            trainB_list = os.path.join(dataset_dir, "trainB.txt")
            a_train_reader = reader_creator(
                image_dir=dataset_dir,
                list_filename=trainA_list,
                batch_size=cfg.batch_size,
                drop_last=cfg.drop_last)
            b_train_reader = reader_creator(
                image_dir=dataset_dir,
                list_filename=trainB_list,
                batch_size=cfg.batch_size,
                drop_last=cfg.drop_last)
            a_reader_test = None
            b_reader_test = None
            if cfg.run_test:
                testA_list = os.path.join(dataset_dir, "testA.txt")
                testB_list = os.path.join(dataset_dir, "testB.txt")
                a_test_reader = reader_creator(
                    image_dir=dataset_dir,
                    list_filename=testA_list,
                    batch_size=1,
                    drop_last=cfg.drop_last)
                b_test_reader = reader_creator(
                    image_dir=dataset_dir,
                    list_filename=testB_list,
                    batch_size=1,
                    drop_last=cfg.drop_last)
                a_reader_test = a_test_reader.get_reader(
                    cfg, mode='test', shuffle=False, return_name=True)
                b_reader_test = b_test_reader.get_reader(
                    cfg, mode='test', shuffle=False, return_name=True)
            if cfg.run_ce:
                batch_num = 1
            else:
                batch_num = max(a_train_reader.len(), b_train_reader.len())
            a_reader = a_train_reader.get_reader(
                cfg, mode='train', shuffle=shuffle)
            b_reader = b_train_reader.get_reader(
                cfg, mode='train', shuffle=shuffle)

        else:
            dataset_dir = os.path.join(cfg.data_dir, cfg.dataset)
            train_list = os.path.join(dataset_dir, cfg.data_list)
            train_reader = reader_creator(
                image_dir=dataset_dir, list_filename=train_list)

    if cfg.model_net == 'cgan':
        from trainer.CGAN import CGAN
        if cfg.dataset != 'mnist':
            raise NotImplementedError('CGAN only support mnist now!')
        model = CGAN(cfg, train_reader)
    elif cfg.model_net == 'dcgan':
        from trainer.DCGAN import DCGAN
        model = DCGAN(cfg, train_reader)
    elif cfg.model_net == 'cyclegan':
        from trainer.CycleGAN import CycleGAN
        model = CycleGAN(cfg, a_reader, b_reader, a_reader_test, b_reader_test,
                         batch_num)
    else:
        pass

    model.build_model()


if __name__ == "__main__":
    cfg = config.parse_args()
    config.print_arguments(cfg)
    assert cfg.load_size >= cfg.crop_size, "Load Size CANNOT less than Crop Size!"
    if cfg.profile:
        if cfg.use_gpu:
            with profiler.profiler('All', 'total', '/tmp/profile') as prof:
                train(cfg)
        else:
            with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                train(cfg)
    else:
        train(cfg)
