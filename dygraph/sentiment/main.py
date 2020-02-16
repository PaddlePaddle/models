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
from utils import ArgumentGroup
from utils import get_cards

parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("checkpoints", str, "checkpoints", "Path to save checkpoints")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 50, "Number of epoches for training.")
train_g.add_arg("save_steps", int, 200,
                "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps", int, 200,
                "The steps interval to evaluate model performance.")
train_g.add_arg("lr", float, 0.002, "The Learning rate value for training.")
train_g.add_arg("padding_size", int, 150,
                "The padding size for input sequences.")

log_g = ArgumentGroup(parser, "logging", "logging related")
log_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")
log_g.add_arg("verbose", bool, False, "Whether to output verbose log")

data_g = ArgumentGroup(parser, "data",
                       "Data paths, vocab paths and data processing options")
data_g.add_arg("data_dir", str, "./senta_data/", "Path to training data.")
data_g.add_arg("vocab_path", str, "./senta_data/word_dict.txt",
               "Vocabulary path.")
data_g.add_arg("vocab_size", int, 33256, "Vocabulary path.")
data_g.add_arg("batch_size", int, 256,
               "Total examples' number in batch for training.")
data_g.add_arg("random_seed", int, 0, "Random seed.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
run_type_g.add_arg("do_train", bool, True, "Whether to perform training.")
run_type_g.add_arg("do_val", bool, True, "Whether to perform evaluation.")
run_type_g.add_arg("do_infer", bool, False, "Whether to perform inference.")
run_type_g.add_arg("profile_steps", int, 60000,
                   "The steps interval to record the performance.")
train_g.add_arg("model_type", str, "bow_net", "Model type of training.")
parser.add_argument("--ce", action="store_true", help="run ce")

args = parser.parse_args()

if args.use_cuda:
    place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    dev_count = fluid.core.get_cuda_device_count()
else:
    place = fluid.CPUPlace()
    dev_count = 1

import paddle.fluid.profiler as profiler
import contextlib


@contextlib.contextmanager
def profile_context(profile=True):
    if profile:
        with profiler.profiler('All', 'total', '/tmp/profile_file'):
            yield
    else:
        yield

if args.ce:
    print("ce mode")
    seed = 90
    np.random.seed(seed)
    fluid.default_startup_program().random_seed = seed
    fluid.default_main_program().random_seed = seed


def train():
    with fluid.dygraph.guard(place):
        if args.ce:
            print("ce mode")
            seed = 90
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
        processor = reader.SentaProcessor(
            data_dir=args.data_dir,
            vocab_path=args.vocab_path,
            random_seed=args.random_seed)
        num_labels = len(processor.get_labels())

        num_train_examples = processor.get_num_examples(phase="train")

        max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

        if not args.ce:
            train_data_generator = processor.data_generator(
                batch_size=args.batch_size,
                phase='train',
                epoch=args.epoch,
                shuffle=True)

            eval_data_generator = processor.data_generator(
                batch_size=args.batch_size,
                phase='dev',
                epoch=args.epoch,
                shuffle=False)
        else:
            train_data_generator = processor.data_generator(
                batch_size=args.batch_size,
                phase='train',
                epoch=args.epoch,
                shuffle=False)

            eval_data_generator = processor.data_generator(
                batch_size=args.batch_size,
                phase='dev',
                epoch=args.epoch,
                shuffle=False)
        if args.model_type == 'cnn_net':
            model = nets.CNN( args.vocab_size, args.batch_size,
                             args.padding_size)
        elif args.model_type == 'bow_net':
            model = nets.BOW( args.vocab_size, args.batch_size,
                             args.padding_size)
        elif args.model_type == 'gru_net':
            model = nets.GRU( args.vocab_size, args.batch_size,
                             args.padding_size)
        elif args.model_type == 'bigru_net':
            model = nets.BiGRU( args.vocab_size, args.batch_size,
                             args.padding_size)
        sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=args.lr,parameter_list=model.parameters())
        steps = 0
        total_cost, total_acc, total_num_seqs = [], [], []
        gru_hidden_data = np.zeros((args.batch_size, 128), dtype='float32')
        ce_time, ce_infor = [], []
        for eop in range(args.epoch):
            time_begin = time.time()
            for batch_id, data in enumerate(train_data_generator()):
                enable_profile = steps > args.profile_steps
                with profile_context(enable_profile):
                    steps += 1
                    doc = to_variable(
                        np.array([
                            np.pad(x[0][0:args.padding_size], (
                                0, args.padding_size - len(x[0][
                                    0:args.padding_size])),
                                   'constant',
                                   constant_values=(args.vocab_size))
                            for x in data
                        ]).astype('int64').reshape(-1))
                    label = to_variable(
                        np.array([x[1] for x in data]).astype('int64').reshape(
                            args.batch_size, 1))
                    model.train()
                    avg_cost, prediction, acc = model(doc, label)
                    avg_cost.backward()
                    np_mask = (doc.numpy() != args.vocab_size).astype('int32')
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
                              (steps,
                               np.sum(total_cost) / np.sum(total_num_seqs),
                               np.sum(total_acc) / np.sum(total_num_seqs),
                               args.skip_steps / used_time))
                        ce_time.append(used_time)
                        ce_infor.append(np.sum(total_acc) / np.sum(total_num_seqs))
                        total_cost, total_acc, total_num_seqs = [], [], []
                        time_begin = time.time()

                    if steps % args.validation_steps == 0:
                        total_eval_cost, total_eval_acc, total_eval_num_seqs = [], [], []
                        model.eval()
                        eval_steps = 0
                        gru_hidden_data = np.zeros((args.batch_size, 128), dtype='float32')
                        for eval_batch_id, eval_data in enumerate(
                                eval_data_generator()):
                            eval_np_doc = np.array([
                                np.pad(x[0][0:args.padding_size],
                                       (0, args.padding_size -
                                        len(x[0][0:args.padding_size])),
                                       'constant',
                                       constant_values=(args.vocab_size))
                                for x in eval_data
                            ]).astype('int64').reshape(-1)
                            eval_label = to_variable(
                                np.array([x[1] for x in eval_data]).astype(
                                    'int64').reshape(args.batch_size, 1))
                            eval_doc = to_variable(eval_np_doc)
                            eval_avg_cost, eval_prediction, eval_acc = model(
                                eval_doc, eval_label)
                            eval_np_mask = (
                                eval_np_doc != args.vocab_size).astype('int32')
                            eval_word_num = np.sum(eval_np_mask)
                            total_eval_cost.append(eval_avg_cost.numpy() *
                                                   eval_word_num)
                            total_eval_acc.append(eval_acc.numpy() *
                                                  eval_word_num)
                            total_eval_num_seqs.append(eval_word_num)

                            eval_steps += 1

                        time_end = time.time()
                        used_time = time_end - time_begin
                        print(
                            "Final validation result: step: %d, ave loss: %f, "
                            "ave acc: %f, speed: %f steps/s" %
                            (steps, np.sum(total_eval_cost) /
                             np.sum(total_eval_num_seqs), np.sum(total_eval_acc)
                             / np.sum(total_eval_num_seqs),
                             eval_steps / used_time))
                        time_begin = time.time()
                        if args.ce:
                            print("kpis\ttrain_loss\t%0.3f" %
                                  (np.sum(total_eval_cost) /
                                   np.sum(total_eval_num_seqs)))
                            print("kpis\ttrain_acc\t%0.3f" %
                                  (np.sum(total_eval_acc) /
                                   np.sum(total_eval_num_seqs)))

                    if steps % args.save_steps == 0:
                        save_path = args.checkpoints+"/"+"save_dir_" + str(steps)
                        print('save model to: ' + save_path)
                        fluid.dygraph.save_dygraph(model.state_dict(),
                                                   save_path)
                if enable_profile:
                    print('save profile result into /tmp/profile_file')
                    return
        if args.ce:
            card_num = get_cards()
            _acc = 0
            _time = 0
            try:
                _time = ce_time[-1]
                _acc = ce_infor[-1]
            except:
                print("ce info error")
            print("kpis\ttrain_duration_card%s\t%s" % (card_num, _time))
            print("kpis\ttrain_acc_card%s\t%f" % (card_num, _acc))


def infer():
    with fluid.dygraph.guard(place):
        processor = reader.SentaProcessor(
            data_dir=args.data_dir,
            vocab_path=args.vocab_path,
            random_seed=args.random_seed)

        infer_data_generator = processor.data_generator(
            batch_size=args.batch_size,
            phase='infer',
            epoch=args.epoch,
            shuffle=False)
        if args.model_type == 'cnn_net':
            model_infer = nets.CNN( args.vocab_size, args.batch_size,
                                   args.padding_size)
        elif args.model_type == 'bow_net':
            model_infer = nets.BOW( args.vocab_size, args.batch_size,
                                   args.padding_size)
        elif args.model_type == 'gru_net':
            model_infer = nets.GRU( args.vocab_size, args.batch_size,
                                   args.padding_size)
        elif args.model_type == 'bigru_net':
            model_infer = nets.BiGRU( args.vocab_size, args.batch_size,
                                   args.padding_size)
        print('Do inferring ...... ')
        restore, _ = fluid.load_dygraph(args.checkpoints)
        model_infer.set_dict(restore)
        model_infer.eval()
        total_acc, total_num_seqs = [], []
        steps = 0
        time_begin = time.time()
        for batch_id, data in enumerate(infer_data_generator()):
            steps += 1
            np_doc = np.array([np.pad(x[0][0:args.padding_size],
                                       (0, args.padding_size -
                                        len(x[0][0:args.padding_size])),
                                       'constant',
                                       constant_values=(args.vocab_size))
                                for x in data
            ]).astype('int64').reshape(-1)
            doc = to_variable(np_doc)
            label = to_variable(
                np.array([x[1] for x in data]).astype('int64').reshape(
                    args.batch_size, 1))
            _, _, acc = model_infer(doc, label)
            mask = (np_doc != args.vocab_size).astype('int32')
            word_num = np.sum(mask)
            total_acc.append(acc.numpy() * word_num)
            total_num_seqs.append(word_num)
        time_end = time.time()
        used_time = time_end - time_begin       
        print("Final infer result: ave acc: %f, speed: %f steps/s" %
              (np.sum(total_acc) / np.sum(total_num_seqs), steps / used_time))


def main():
    if args.do_train:
        train()
    elif args.do_infer:
        infer()


if __name__ == '__main__':
    print(args)
    main()
