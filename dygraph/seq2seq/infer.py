# -*- coding: utf-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import os
import random
import logging
import math
import io
import paddle
import paddle.fluid as fluid

import reader

import sys
line_tok = '\n'
space_tok = ' '
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
    line_tok = u'\n'
    space_tok = u' '

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

from args import *
import logging
import pickle

from attention_model import AttentionModel
from base_model import BaseModel


def infer():
    args = parse_args()

    num_layers = args.num_layers
    src_vocab_size = args.src_vocab_size
    tar_vocab_size = args.tar_vocab_size
    batch_size = args.batch_size
    dropout = args.dropout
    init_scale = args.init_scale
    max_grad_norm = args.max_grad_norm
    hidden_size = args.hidden_size
    # inference process

    print("src", src_vocab_size)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        # dropout type using upscale_in_train, dropout can be remove in inferecen
        # So we can set dropout to 0
        if args.attention:
            model = AttentionModel(
                hidden_size,
                src_vocab_size,
                tar_vocab_size,
                batch_size,
                beam_size = args.beam_size,
                num_layers=num_layers,
                init_scale=init_scale,
                dropout=0.0,
                mode='beam_search')
        else:
            model = BaseModel(
                hidden_size,
                src_vocab_size,
                tar_vocab_size,
                batch_size,
                beam_size = args.beam_size,
                num_layers=num_layers,
                init_scale=init_scale,
                dropout=0.0,
                mode='beam_search')

        source_vocab_file = args.vocab_prefix + "." + args.src_lang
        infer_file = args.infer_file

        infer_data = reader.raw_mono_data(source_vocab_file, infer_file)

        def prepare_input(batch, epoch_id=0):
                src_ids, src_mask, tar_ids, tar_mask = batch
                res = {}
                src_ids = src_ids.reshape((src_ids.shape[0], src_ids.shape[1]))
                in_tar = tar_ids[:, :-1]
                label_tar = tar_ids[:, 1:]

                in_tar = in_tar.reshape((in_tar.shape[0], in_tar.shape[1]))
                label_tar = label_tar.reshape(
                    (label_tar.shape[0], label_tar.shape[1], 1))
                inputs = [src_ids, in_tar, label_tar, src_mask, tar_mask]
                return inputs, np.sum(tar_mask)

        dir_name = args.reload_model
        print("dir name", dir_name)
        state_dict, _ = fluid.dygraph.load_dygraph(dir_name)
        model.set_dict(state_dict)
        model.eval()

        train_data_iter = reader.get_data_iter(infer_data, batch_size, mode='infer')

        tar_id2vocab = []
        tar_vocab_file = args.vocab_prefix + "." + args.tar_lang
        with io.open(tar_vocab_file, "r", encoding='utf-8') as f:
            for line in f.readlines():
                tar_id2vocab.append(line.strip())

        infer_output_file = args.infer_output_file
        infer_output_dir = infer_output_file.split('/')[0]
        if not os.path.exists(infer_output_dir):
            os.mkdir(infer_output_dir)

        with io.open(infer_output_file, 'w', encoding='utf-8') as out_file:

            for batch_id, batch in enumerate(train_data_iter):
                input_data_feed, word_num = prepare_input(batch, epoch_id=0)
                # import ipdb; ipdb.set_trace()
                outputs = model(input_data_feed)
                for i in range(outputs.shape[0]):
                    ins = outputs[i].numpy()
                    res = [tar_id2vocab[int(e)] for e in ins[:, 0].reshape(-1)]
                    new_res = []
                    for ele in res:
                        if ele == "</s>":
                            break
                        new_res.append(ele)

                    out_file.write(space_tok.join(new_res))
                    out_file.write(line_tok)


def check_version():
    """
    Log error and exit when the installed version of paddlepaddle is
    not satisfied.
    """
    err = "PaddlePaddle version 1.6 or higher is required, " \
          "or a suitable develop version is satisfied as well. \n" \
          "Please make sure the version is good with your code." \

    try:
        fluid.require_version('1.6.0')
    except Exception as e:
        logger.error(err)
        sys.exit(1)


if __name__ == '__main__':
    check_version()
    infer()
