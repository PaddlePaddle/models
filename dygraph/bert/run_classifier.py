#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""BERT fine-tuning in Paddle Dygraph Mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import sys
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf8')
import ast
import time
import argparse
import numpy as np
import multiprocessing
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
import reader.cls as reader
from model.bert import BertConfig
from model.cls import ClsModelLayer
from optimization import Optimizer
from utils.args import ArgumentGroup, print_arguments, check_cuda
from utils.init import init_from_static_model

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("bert_config_path",      str,  "./config/bert_config.json",  "Path to the json file for bert model config.")
model_g.add_arg("init_checkpoint",       str,  None,                         "Init checkpoint to resume training from.")
model_g.add_arg("init_pretraining_params",  str,  None,
                "Init pre-training params which preforms fine-tuning from. If the "
                "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
model_g.add_arg("checkpoints",           str,  "checkpoints",                "Path to save checkpoints.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch",             int,    100,     "Number of epoches for training.")
train_g.add_arg("learning_rate",     float,  0.0001,  "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler",      str,    "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay",      float,  0.01,    "Weight decay rate for L2 regularizer.")
train_g.add_arg("warmup_proportion",     float,  0.1,                         "Proportion of training steps to perform linear learning rate warmup for.")
train_g.add_arg("save_steps",        int,    10000,   "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps",  int,    1000,    "The steps interval to evaluate model performance.")
train_g.add_arg("loss_scaling",      float,  1.0,
                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")

log_g = ArgumentGroup(parser,     "logging", "logging related.")
log_g.add_arg("skip_steps",          int,    10,    "The steps interval to print loss.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("data_dir",            str,  None,       "Path to training data.")
data_g.add_arg("vocab_path",          str,  None,       "Vocabulary path.")
data_g.add_arg("max_seq_len",         int,  512,                   "Tokens' number of the longest seqence allowed.")
data_g.add_arg("batch_size",          int,  32,
               "The total number of examples in one batch for training, see also --in_tokens.")
data_g.add_arg("in_tokens",           bool, False,
               "If set, the batch size will be the maximum number of tokens in one batch. "
               "Otherwise, it will be the maximum number of examples in one batch.")
data_g.add_arg("do_lower_case", bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
data_g.add_arg("random_seed",   int,  5512,     "Random seed.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",                     bool,   True,   "If set, use GPU for training.")
run_type_g.add_arg("shuffle",                      bool,   True,  "")
run_type_g.add_arg("task_name",                    str,    None,
                   "The name of task to perform fine-tuning, should be in {'xnli', 'mnli', 'cola', 'mrpc'}.")
run_type_g.add_arg("do_train",                     bool,   True,  "Whether to perform training.")
run_type_g.add_arg("do_test",                      bool,   False,  "Whether to perform evaluation on test data set.")
run_type_g.add_arg("use_data_parallel", bool, False,  "The flag indicating whether to shuffle instances in each pass.")
run_type_g.add_arg("enable_ce", bool, False,	 help="The flag indicating whether to run the task for continuous evaluation.")

args = parser.parse_args()

def create_data(batch):
    """
    convert data to variable
    """
    src_ids = to_variable(batch[0], "src_ids")
    position_ids = to_variable(batch[1], "position_ids")
    sentence_ids = to_variable(batch[2], "sentence_ids")
    input_mask = to_variable(batch[3], "input_mask")
    labels = to_variable(batch[4], "labels")
    labels.stop_gradient = True
    return src_ids, position_ids, sentence_ids, input_mask, labels

if args.use_cuda:
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
    dev_count = fluid.core.get_cuda_device_count()
else:
    place = fluid.CPUPlace()
    dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))


def train(args):
    bert_config = BertConfig(args.bert_config_path)
    bert_config.print_config()

    if not (args.do_train or args.do_test):
        raise ValueError("For args `do_train`, `do_test`, at "
                        "least one of them must be True.")

    trainer_count = fluid.dygraph.parallel.Env().nranks

    task_name = args.task_name.lower()
    processors = {
        'xnli': reader.XnliProcessor,
        'cola': reader.ColaProcessor,
        'mrpc': reader.MrpcProcessor,
        'mnli': reader.MnliProcessor,
    }

    processor = processors[task_name](data_dir=args.data_dir,
                                      vocab_path=args.vocab_path,
                                      max_seq_len=args.max_seq_len,
                                      do_lower_case=args.do_lower_case,
                                      in_tokens=args.in_tokens,
                                      random_seed=args.random_seed)
    num_labels = len(processor.get_labels())
    shuffle_seed = 1 if trainer_count > 1 else None

    train_data_generator = processor.data_generator(
                                      batch_size=args.batch_size,
                                      phase='train',
                                      epoch=args.epoch,
                                      dev_count=trainer_count,
                                      shuffle=args.shuffle,
                                      shuffle_seed=shuffle_seed)
    num_train_examples = processor.get_num_examples(phase='train')
    max_train_steps = args.epoch * num_train_examples // args.batch_size // trainer_count
    warmup_steps = int(max_train_steps * args.warmup_proportion)

    print("Device count: %d" % dev_count)
    print("Trainer count: %d" % trainer_count)
    print("Num train examples: %d" % num_train_examples)
    print("Max train steps: %d" % max_train_steps)
    print("Num warmup steps: %d" % warmup_steps)

    with fluid.dygraph.guard(place):

        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()

        cls_model = ClsModelLayer(
                            args,
                            bert_config,
                            num_labels,
                            is_training=True,
                            return_pooled_out=True)

        optimizer = Optimizer(
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    model_cls=cls_model,
                    weight_decay=args.weight_decay,
                    scheduler=args.lr_scheduler,
                    loss_scaling=args.loss_scaling,
                    parameter_list=cls_model.parameters())

        if args.init_pretraining_params:
            print("Load pre-trained model from %s" % args.init_pretraining_params)
            init_from_static_model(args.init_pretraining_params, cls_model, bert_config)

        if args.use_data_parallel:
            cls_model = fluid.dygraph.parallel.DataParallel(cls_model, strategy)
            train_data_generator = fluid.contrib.reader.distributed_batch_reader(train_data_generator)

        steps = 0
        time_begin = time.time()

        ce_time = []
        ce_acc = []
        for batch in train_data_generator():
            data_ids = create_data(batch)
            loss, accuracy, num_seqs = cls_model(data_ids)

            optimizer.optimization(loss, use_data_parallel = args.use_data_parallel, model = cls_model)
            cls_model.clear_gradients()

            if steps != 0 and steps % args.skip_steps == 0:
                time_end = time.time()
                used_time = time_end - time_begin
                current_example, current_epoch = processor.get_train_progress()
                localtime = time.asctime(time.localtime(time.time()))
                print("%s, epoch: %s, steps: %s, dy_graph loss: %f, acc: %f, speed: %f steps/s" % (localtime, current_epoch, steps, loss.numpy(), accuracy.numpy(), args.skip_steps / used_time))
                ce_time.append(used_time)
                ce_acc.append(accuracy.numpy())
                time_begin = time.time()

            if steps != 0 and steps % args.save_steps == 0 and fluid.dygraph.parallel.Env().local_rank == 0:
                save_path = os.path.join(args.checkpoints, "steps" + "_" + str(steps))
                fluid.save_dygraph(
                    cls_model.state_dict(),
                    save_path)
                fluid.save_dygraph(
                    optimizer.optimizer.state_dict(),
                    save_path)
                print("Save model parameters and optimizer status at %s" % save_path)

            steps += 1

        if fluid.dygraph.parallel.Env().local_rank == 0:
            save_path = os.path.join(args.checkpoints, "final")
            fluid.save_dygraph(
                cls_model.state_dict(),
                save_path)
            fluid.save_dygraph(
                optimizer.optimizer.state_dict(),
                save_path)
            print("Save model parameters and optimizer status at %s" % save_path)

        if args.enable_ce:
            _acc = 0
            _time = 0
            try:
                _time = ce_time[-1]
                _acc = ce_acc[-1]
            except:
                print("ce info error")
            print("kpis\ttrain_duration_card%s\t%s" % (dev_count, _time))
            print("kpis\ttrain_acc_card%s\t%f" % (dev_count, _acc))

        return cls_model

def predict(args, cls_model = None):

    bert_config = BertConfig(args.bert_config_path)
    bert_config.print_config()

    task_name = args.task_name.lower()
    processors = {
        'xnli': reader.XnliProcessor,
        'cola': reader.ColaProcessor,
        'mrpc': reader.MrpcProcessor,
        'mnli': reader.MnliProcessor,
    }

    processor = processors[task_name](data_dir=args.data_dir,
            vocab_path=args.vocab_path,
            max_seq_len=args.max_seq_len,
            do_lower_case=args.do_lower_case,
            in_tokens=False)

    test_data_generator = processor.data_generator(
                                batch_size=args.batch_size,
                                phase='dev',
                                epoch=1,
                                shuffle=False)

    num_labels = len(processor.get_labels())

    with fluid.dygraph.guard(place):
        if cls_model is None:
            cls_model = ClsModelLayer(
                args,
                bert_config,
                num_labels,
                is_training=False,
                return_pooled_out=True)

            #restore the model
            save_path = os.path.join(args.checkpoints, "final")
            print("Load params from %s" % save_path)
            model_dict,_ = fluid.load_dygraph(save_path)
            cls_model.load_dict(model_dict)

        print('Do predicting ...... ')
        cls_model.eval()

        total_cost, total_acc, total_num_seqs = [], [], []

        for batch in test_data_generator():
            data_ids = create_data(batch)
            np_loss, np_acc, np_num_seqs = cls_model(data_ids)

            np_loss = np_loss.numpy()
            np_acc = np_acc.numpy()
            np_num_seqs = np_num_seqs.numpy()

            total_cost.extend(np_loss * np_num_seqs)
            total_acc.extend(np_acc * np_num_seqs)
            total_num_seqs.extend(np_num_seqs)

        print("[evaluation] average acc: %f" % (np.sum(total_acc) / np.sum(total_num_seqs)))


if __name__ == '__main__':

    print_arguments(args)
    check_cuda(args.use_cuda)

    if args.do_train:
        cls_model = train(args)
        if args.do_test:
            predict(args, cls_model = cls_model)

    elif args.do_test:
        predict(args)
