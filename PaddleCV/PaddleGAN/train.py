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
import trainer


def train(cfg):

    MODELS = [
        "CGAN", "DCGAN", "Pix2pix", "CycleGAN", "StarGAN", "AttGAN", "STGAN",
        "SPADE"
    ]
    if cfg.model_net not in MODELS:
        raise NotImplementedError("{} is not support!".format(cfg.model_net))

    reader = data_reader(cfg)

    if cfg.model_net in ['CycleGAN']:
        a_reader, b_reader, a_reader_test, b_reader_test, batch_num, a_id2name, b_id2name = reader.make_data(
        )
    else:
        if cfg.dataset in ['mnist']:
            train_reader = reader.make_data()
        else:
            train_reader, test_reader, batch_num, id2name = reader.make_data()

    if cfg.model_net in ['CGAN', 'DCGAN']:
        if cfg.dataset != 'mnist':
            raise NotImplementedError("CGAN/DCGAN only support MNIST now!")
        model = trainer.__dict__[cfg.model_net](cfg, train_reader)
    elif cfg.model_net in ['CycleGAN']:
        model = trainer.__dict__[cfg.model_net](cfg, a_reader, b_reader,
                                                a_reader_test, b_reader_test,
                                                batch_num, a_id2name, b_id2name)
    else:
        model = trainer.__dict__[cfg.model_net](cfg, train_reader, test_reader,
                                                batch_num, id2name)

    model.build_model()


if __name__ == "__main__":
    cfg = config.parse_args()
    config.print_arguments(cfg)
    utility.check_gpu(cfg.use_gpu)
    utility.check_version()
    if cfg.profile:
        if cfg.use_gpu:
            with fluid.profiler.profiler('All', 'total',
                                         cfg.profiler_path) as prof:
                train(cfg)
        else:
            with fluid.profiler.profiler("CPU", sorted_key='total') as cpuprof:
                train(cfg)
    else:
        train(cfg)
