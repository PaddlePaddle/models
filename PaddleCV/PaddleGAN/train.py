#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import division
from __future__ import print_function

from util import config, utility
from data_reader import data_reader
import os
import sys
import six
import time
import numpy as np
import paddle
import paddle.fluid as fluid


def train(cfg):
    reader = data_reader(cfg)
    if cfg.model_net == 'CycleGAN':
        a_reader, b_reader, a_reader_test, b_reader_test, batch_num = reader.make_data(
        )
    elif cfg.model_net == 'Pix2pix':
        train_reader, test_reader, batch_num = reader.make_data()
    elif cfg.model_net == 'StarGAN':
        train_reader, test_reader, batch_num = reader.make_data()
    else:
        if cfg.dataset == 'mnist':
            train_reader = reader.make_data()
        else:
            train_reader, test_reader, batch_num = reader.make_data()

    if cfg.model_net == 'CGAN':
        from trainer.CGAN import CGAN
        if cfg.dataset != 'mnist':
            raise NotImplementedError('CGAN only support mnist now!')
        model = CGAN(cfg, train_reader)
    elif cfg.model_net == 'DCGAN':
        from trainer.DCGAN import DCGAN
        if cfg.dataset != 'mnist':
            raise NotImplementedError('DCGAN only support mnist now!')
        model = DCGAN(cfg, train_reader)
    elif cfg.model_net == 'CycleGAN':
        from trainer.CycleGAN import CycleGAN
        model = CycleGAN(cfg, a_reader, b_reader, a_reader_test, b_reader_test,
                         batch_num)
    elif cfg.model_net == 'Pix2pix':
        from trainer.Pix2pix import Pix2pix
        model = Pix2pix(cfg, train_reader, test_reader, batch_num)
    elif cfg.model_net == 'StarGAN':
        from trainer.StarGAN import StarGAN
        model = StarGAN(cfg, train_reader, test_reader, batch_num)
    elif cfg.model_net == 'AttGAN':
        from trainer.AttGAN import AttGAN
        model = AttGAN(cfg, train_reader, test_reader, batch_num)
    elif cfg.model_net == 'STGAN':
        from trainer.STGAN import STGAN
        model = STGAN(cfg, train_reader, test_reader, batch_num)
    else:
        pass

    model.build_model()


if __name__ == "__main__":
    cfg = config.parse_args()
    config.print_arguments(cfg)
    utility.check_gpu(cfg.use_gpu)
    #assert cfg.load_size >= cfg.crop_size, "Load Size CANNOT less than Crop Size!"
    if cfg.profile:
        if cfg.use_gpu:
            with profiler.profiler('All', 'total', '/tmp/profile') as prof:
                train(cfg)
        else:
            with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                train(cfg)
    else:
        train(cfg)
