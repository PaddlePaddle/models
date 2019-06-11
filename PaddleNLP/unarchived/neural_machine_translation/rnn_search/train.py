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
import time
import os

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor
from paddle.fluid.contrib.decoder.beam_search_decoder import *

from args import *
import attention_model
import no_attention_model


def train():
    args = parse_args()

    if args.enable_ce:
        framework.default_startup_program().random_seed = 111

    # Training process
    if args.no_attention:
        avg_cost, feed_order = no_attention_model.seq_to_seq_net(
            args.embedding_dim,
            args.encoder_size,
            args.decoder_size,
            args.dict_size,
            args.dict_size,
            False,
            beam_size=args.beam_size,
            max_length=args.max_length)
    else:
        avg_cost, feed_order = attention_model.seq_to_seq_net(
            args.embedding_dim,
            args.encoder_size,
            args.decoder_size,
            args.dict_size,
            args.dict_size,
            False,
            beam_size=args.beam_size,
            max_length=args.max_length)

    # clone from default main program and use it as the validation program
    main_program = fluid.default_main_program()
    inference_program = fluid.default_main_program().clone()

    optimizer = fluid.optimizer.Adam(
        learning_rate=args.learning_rate,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=1e-5))

    optimizer.minimize(avg_cost)

    # Disable shuffle for Continuous Evaluation only
    if not args.enable_ce:
        train_batch_generator = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.wmt14.train(args.dict_size), buf_size=1000),
            batch_size=args.batch_size,
            drop_last=False)

        test_batch_generator = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.wmt14.test(args.dict_size), buf_size=1000),
            batch_size=args.batch_size,
            drop_last=False)
    else:
        train_batch_generator = paddle.batch(
            paddle.dataset.wmt14.train(args.dict_size),
            batch_size=args.batch_size,
            drop_last=False)

        test_batch_generator = paddle.batch(
            paddle.dataset.wmt14.test(args.dict_size),
            batch_size=args.batch_size,
            drop_last=False)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    feed_list = [
        main_program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(feed_list, place)

    def validation():
        # Use test set as validation each pass
        total_loss = 0.0
        count = 0
        val_feed_list = [
            inference_program.global_block().var(var_name)
            for var_name in feed_order
        ]
        val_feeder = fluid.DataFeeder(val_feed_list, place)

        for batch_id, data in enumerate(test_batch_generator()):
            val_fetch_outs = exe.run(inference_program,
                                     feed=val_feeder.feed(data),
                                     fetch_list=[avg_cost],
                                     return_numpy=False)

            total_loss += np.array(val_fetch_outs[0])[0]
            count += 1

        return total_loss / count

    for pass_id in range(1, args.pass_num + 1):
        pass_start_time = time.time()
        words_seen = 0
        for batch_id, data in enumerate(train_batch_generator()):
            words_seen += len(data) * 2

            fetch_outs = exe.run(framework.default_main_program(),
                                 feed=feeder.feed(data),
                                 fetch_list=[avg_cost])

            avg_cost_train = np.array(fetch_outs[0])
            print('pass_id=%d, batch_id=%d, train_loss: %f' %
                  (pass_id, batch_id, avg_cost_train))
            # This is for continuous evaluation only
            if args.enable_ce and batch_id >= 100:
                break

        pass_end_time = time.time()
        test_loss = validation()
        time_consumed = pass_end_time - pass_start_time
        words_per_sec = words_seen / time_consumed
        print("pass_id=%d, test_loss: %f, words/s: %f, sec/pass: %f" %
              (pass_id, test_loss, words_per_sec, time_consumed))

        # This log is for continuous evaluation only
        if args.enable_ce:
            print("kpis\ttrain_cost\t%f" % avg_cost_train)
            print("kpis\ttest_cost\t%f" % test_loss)
            print("kpis\ttrain_duration\t%f" % time_consumed)

        if pass_id % args.save_interval == 0:
            model_path = os.path.join(args.save_dir, str(pass_id))
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            fluid.io.save_persistables(
                executor=exe,
                dirname=model_path,
                main_program=framework.default_main_program())


if __name__ == '__main__':
    train()
