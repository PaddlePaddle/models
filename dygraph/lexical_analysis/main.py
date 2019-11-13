# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import nets
import reader
import utils


def train(args, place):
    with fluid.dygraph.guard(place):

        dataset = reader.Dataset(args)
        num_train_examples = dataset.get_num_examples(args.train_data)

        max_train_steps = args.epoch * num_train_examples // args.batch_size

        #define reader
        train_processor = reader.LACProcessor(args, args.train_data,
                                              args.word_dict_path)
        test_processor = dataset.file_reader(args.test_data, mode="test")

        #define network
        model = nets.LAC("lac_net", args, dataset.vocab_size, args.batch_size,
                         args.max_seq_lens)

        sgd_optimizer = fluid.optimizer.Adagrad(
            learning_rate=args.base_learning_rate)
        steps = 0
        total_cost, total_acc, total_num_seqs = [], [], []
        for eop in range(args.epoch):
            time_begin = time.time()
            for data in train_processor.data_generator("train")():
                steps += 1
                doc = to_variable(
                    np.array([
                        np.pad(x[0][0:args.max_seq_lens], (
                            0, args.max_seq_lens - len(x[0][
                                0:args.max_seq_lens])),
                               'constant',
                               constant_values=(dataset.vocab_size))
                        for x in data
                    ]).astype('int64').reshape(-1, 1))

                seq_lens = to_variable(
                    np.array([len(x[0]) for x in data]).astype('int64'))
                targets = to_variable(
                    np.array([
                        np.pad(x[1][0:args.max_seq_lens], (
                            0, args.max_seq_lens - len(x[1][
                                0:args.max_seq_lens])),
                               'constant',
                               constant_values=(dataset.num_labels))
                        for x in data
                    ]).astype('int64'))

                model.train()
                avg_cost, prediction, acc = model(doc, targets, seq_lens)
                avg_cost.backward()
                np_mask = (doc.numpy() != dataset.vocab_size).astype('int32')
                word_num = np.sum(np_mask)
                sgd_optimizer.minimize(avg_cost)
                model.clear_gradients()
                total_cost.append(avg_cost.numpy() * word_num)
                total_acc.append(acc.numpy() * word_num)
                total_num_seqs.append(word_num)

                if steps % args.skip_steps == 0:
                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("step: %d, ave loss: %f, "
                          "ave acc: %f, speed: %f steps/s" %
                          (steps, np.sum(total_cost) / np.sum(total_num_seqs),
                           np.sum(total_acc) / np.sum(total_num_seqs),
                           args.skip_steps / used_time))
                    total_cost, total_acc, total_num_seqs = [], [], []
                    time_begin = time.time()

                if steps % args.validation_steps == 0:
                    total_eval_cost, total_eval_acc, total_eval_num_seqs = [], [], []
                    model.eval()
                    eval_steps = 0
                    for data in train_processor.data_generator("train")():
                        steps += 1
                        eval_doc = to_variable(
                            np.array([
                                np.pad(x[0][0:args.max_seq_lens], (
                                    0, args.max_seq_lens - len(x[0][
                                        0:args.max_seq_lens])),
                                       'constant',
                                       constant_values=(dataset.vocab_size))
                                for x in data
                            ]).astype('int64').reshape(-1, 1))

                        eval_seq_lens = to_variable(
                            np.array([len(x[0]) for x in data]).astype('int64')
                            .reshape(args.batch_size, 1))

                        eval_targets = to_variable(
                            np.array([
                                np.pad(x[1][0:args.max_seq_lens], (
                                    0, args.max_seq_lens - len(x[1][
                                        0:args.max_seq_lens])),
                                       'constant',
                                       constant_values=(dataset.num_labels))
                                for x in data
                            ]).astype('int64'))

                        eval_avg_cost, eval_prediction, eval_acc = model(
                            eval_doc, eval_targets, eval_seq_lens)
                        eval_np_mask = (
                            eval_np_doc != dataset.vocab_size).astype('int32')
                        eval_word_num = np.sum(eval_np_mask)
                        total_eval_cost.append(eval_avg_cost.numpy() *
                                               eval_word_num)
                        total_eval_acc.append(eval_acc.numpy() * eval_word_num)
                        total_eval_num_seqs.append(eval_word_num)
                        eval_steps += 1

                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("Final validation result: step: %d, ave loss: %f, "
                          "ave acc: %f, speed: %f steps/s" %
                          (steps, np.sum(total_eval_cost) /
                           np.sum(total_eval_num_seqs), np.sum(total_eval_acc) /
                           np.sum(total_eval_num_seqs), eval_steps / used_time))
                    time_begin = time.time()
                    if args.ce:
                        print("kpis\ttrain_loss\t%0.3f" %
                              (np.sum(total_eval_cost) /
                               np.sum(total_eval_num_seqs)))
                        print("kpis\ttrain_acc\t%0.3f" %
                              (np.sum(total_eval_acc) /
                               np.sum(total_eval_num_seqs)))

                    if steps % args.save_steps == 0:
                        save_path = "save_dir_" + str(steps)
                        print('save model to: ' + save_path)
                        fluid.dygraph.save_dygraph(model.state_dict(),
                                                   save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    utils.load_yaml(parser, 'args.yaml')
    args = parser.parse_args()
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
    dev_count = 1
    print(args)
    train(args, place)
