#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import os
import six

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor
from paddle.fluid.contrib.decoder.beam_search_decoder import *

from args import *
import attention_model
import no_attention_model


def infer():
    args = parse_args()

    # Inference
    if args.no_attention:
        translation_ids, translation_scores, feed_order = \
            no_attention_model.seq_to_seq_net(
            args.embedding_dim,
            args.encoder_size,
            args.decoder_size,
            args.dict_size,
            args.dict_size,
            True,
            beam_size=args.beam_size,
            max_length=args.max_length)
    else:
        translation_ids, translation_scores, feed_order = \
            attention_model.seq_to_seq_net(
            args.embedding_dim,
            args.encoder_size,
            args.decoder_size,
            args.dict_size,
            args.dict_size,
            True,
            beam_size=args.beam_size,
            max_length=args.max_length)

    test_batch_generator = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.test(args.dict_size), buf_size=1000),
        batch_size=args.batch_size,
        drop_last=False)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    model_path = os.path.join(args.save_dir, str(args.pass_num))
    fluid.io.load_persistables(
        executor=exe,
        dirname=model_path,
        main_program=framework.default_main_program())

    src_dict, trg_dict = paddle.dataset.wmt14.get_dict(args.dict_size)

    feed_list = [
        framework.default_main_program().global_block().var(var_name)
        for var_name in feed_order[0:1]
    ]
    feeder = fluid.DataFeeder(feed_list, place)

    for batch_id, data in enumerate(test_batch_generator()):
        # The value of batch_size may vary in the last batch
        batch_size = len(data)

        # Setup initial ids and scores lod tensor
        init_ids_data = np.array([0 for _ in range(batch_size)], dtype='int64')
        init_scores_data = np.array(
            [1. for _ in range(batch_size)], dtype='float32')
        init_ids_data = init_ids_data.reshape((batch_size, 1))
        init_scores_data = init_scores_data.reshape((batch_size, 1))
        init_recursive_seq_lens = [1] * batch_size
        init_recursive_seq_lens = [
            init_recursive_seq_lens, init_recursive_seq_lens
        ]
        init_ids = fluid.create_lod_tensor(init_ids_data,
                                           init_recursive_seq_lens, place)
        init_scores = fluid.create_lod_tensor(init_scores_data,
                                              init_recursive_seq_lens, place)

        # Feed dict for inference
        feed_dict = feeder.feed([[x[0]] for x in data])
        feed_dict['init_ids'] = init_ids
        feed_dict['init_scores'] = init_scores

        fetch_outs = exe.run(framework.default_main_program(),
                             feed=feed_dict,
                             fetch_list=[translation_ids, translation_scores],
                             return_numpy=False)

        # Split the output words by lod levels
        lod_level_1 = fetch_outs[0].lod()[1]
        token_array = np.array(fetch_outs[0])
        result = []
        for i in six.moves.xrange(len(lod_level_1) - 1):
            sentence_list = [
                trg_dict[token]
                for token in token_array[lod_level_1[i]:lod_level_1[i + 1]]
            ]
            sentence = " ".join(sentence_list[1:-1])
            result.append(sentence)
        lod_level_0 = fetch_outs[0].lod()[0]
        paragraphs = [
            result[lod_level_0[i]:lod_level_0[i + 1]]
            for i in six.moves.xrange(len(lod_level_0) - 1)
        ]

        for paragraph in paragraphs:
            print(paragraph)


if __name__ == '__main__':
    infer()
