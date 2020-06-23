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
"""BERT pretraining in Paddle Dygraph Mode"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import sys
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf8')

import os
import time
import argparse
import numpy as np
import multiprocessing

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable

from reader.pretraining import DataReader
from model.bert import PretrainModelLayer, BertConfig
from optimization import Optimizer
from utils.args import ArgumentGroup, print_arguments, check_cuda
from utils.init import init_checkpoint, init_pretraining_params, init_from_static_model

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("bert_config_path",      str,  "./config/bert_config.json",  "Path to the json file for bert model config.")
model_g.add_arg("init_checkpoint",       str,  None,                         "Init checkpoint to resume training from.")
model_g.add_arg("checkpoints",           str,  "checkpoints",                "Path to save checkpoints.")
model_g.add_arg("weight_sharing",        bool, True,                         "If set, share weights between word embedding and masked lm.")
model_g.add_arg("generate_neg_sample",   bool, True,                         "If set, randomly generate negtive samples by positive samples.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch",             int,    100,     "Number of epoches for training.")
train_g.add_arg("learning_rate",     float,  0.0001,  "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler",      str,    "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay",      float,  0.01,    "Weight decay rate for L2 regularizer.")
train_g.add_arg("num_train_steps",   int,    1000000, "Total steps to perform pretraining.")
train_g.add_arg("warmup_steps",      int,    4000,    "Total steps to perform warmup when pretraining.")
train_g.add_arg("save_steps",        int,    10000,   "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps",  int,    1000,    "The steps interval to evaluate model performance.")
train_g.add_arg("use_fp16",          bool,   False,   "Whether to use fp16 mixed precision training.")
train_g.add_arg("use_dynamic_loss_scaling",    bool,   True,   "Whether to use dynamic loss scaling in mixed precision training.")
train_g.add_arg("init_loss_scaling",           float,  2**32,
                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")
train_g.add_arg("loss_scaling",      float,  1.0,
                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")
train_g.add_arg("incr_every_n_steps",          int,    1000,   "Increases loss scaling every n consecutive.")
train_g.add_arg("decr_every_n_nan_or_inf",     int,    2,
                "Decreases loss scaling every n accumulated steps with nan or inf gradients.")
train_g.add_arg("incr_ratio",                  float,  2.0,
                "The multiplier to use when increasing the loss scaling.")
train_g.add_arg("decr_ratio",                  float,  0.8,
                "The less-than-one-multiplier to use when decreasing.")

log_g = ArgumentGroup(parser,     "logging", "logging related.")
log_g.add_arg("skip_steps",          int,    10,    "The steps interval to print loss.")
log_g.add_arg("verbose",             bool,   False, "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("data_dir",            str,  "./data/train/",       "Path to training data.")
data_g.add_arg("validation_set_dir",  str,  "./data/validation/",  "Path to validation data.")
data_g.add_arg("test_set_dir",        str,  None,                  "Path to test data.")
data_g.add_arg("vocab_path",          str,  "./config/vocab.txt",  "Vocabulary path.")
data_g.add_arg("max_seq_len",         int,  512,                   "Tokens' number of the longest seqence allowed.")
data_g.add_arg("batch_size",          int,  8192,
               "The total number of examples in one batch for training, see also --in_tokens.")
data_g.add_arg("in_tokens",           bool, True,
               "If set, the batch size will be the maximum number of tokens in one batch. "
               "Otherwise, it will be the maximum number of examples in one batch.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("is_distributed",               bool,   False,  "If set, then start distributed training.")
run_type_g.add_arg("use_cuda",                     bool,   True,   "If set, use GPU for training.")
run_type_g.add_arg("use_fast_executor",            bool,   False,  "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("num_iteration_per_drop_scope", int,    1,      "Ihe iteration intervals to clean up temporary variables.")
run_type_g.add_arg("do_test",                      bool,   False,  "Whether to perform evaluation on test data set.")
run_type_g.add_arg("use_data_parallel",            bool,   False,  "The flag indicating whether to shuffle instances in each pass.")

args = parser.parse_args()
# yapf: enable.


def create_data(batch):
    """
    convert data to variable
    """
    src_ids = to_variable(batch[0], "src_ids")
    position_ids = to_variable(batch[1], "position_ids")
    sentence_ids = to_variable(batch[2], "sentence_ids")
    input_mask = to_variable(batch[3], "input_mask")
    mask_label = to_variable(batch[4], "mask_label")
    mask_pos = to_variable(batch[5], "mask_pos")
    labels = to_variable(batch[6], "labels")
    labels.stop_gradient = True
    return src_ids, position_ids, sentence_ids, input_mask, mask_label, mask_pos, labels


def predict_wrapper(pretrained_bert,
                    data_loader=None):
    cost = 0
    lm_cost = 0
    lm_acc = 0
    acc = 0
    steps = 0
    time_begin = time.time()
    try:
        for batch in data_loader():
            steps += 1
            (src_ids, pos_ids, sent_ids, input_mask, mask_label,
             mask_pos, labels) = create_data(batch)
            each_lm_acc, each_next_acc, each_mask_lm_cost, each_total_cost = pretrained_bert(
                src_ids, pos_ids, sent_ids, input_mask,
                mask_label, mask_pos, labels)
            lm_acc += each_lm_acc.numpy()
            acc += each_next_acc.numpy()
            lm_cost += each_mask_lm_cost.numpy()
            cost += each_total_cost.numpy()
    except fluid.core.EOFException:
        data_loader.reset()

    used_time = time.time() - time_begin
    return cost, lm_cost, lm_acc, acc, steps, (steps / used_time)


def train(args):
    print("pretraining start")
    bert_config = BertConfig(args.bert_config_path)
    bert_config.print_config()

    if args.use_cuda:
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get("CPU_NUM", multiprocessing.cpu_count()))

    trainer_count = fluid.dygraph.parallel.Env().nranks

    data_reader = DataReader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        in_tokens=args.in_tokens,
        vocab_path=args.vocab_path,
        voc_size=bert_config['vocab_size'],
        epoch=args.epoch,
        max_seq_len=args.max_seq_len,
        generate_neg_sample=args.generate_neg_sample)
    batch_generator = data_reader.data_generator()
    if args.validation_set_dir and args.validation_set_dir != "":
        val_data_reader = DataReader(
            data_dir=args.validation_set_dir,
            batch_size=args.batch_size,
            in_tokens=args.in_tokens,
            vocab_path=args.vocab_path,
            voc_size=bert_config['vocab_size'],
            shuffle_files=False,
            epoch=1,
            max_seq_len=args.max_seq_len,
            is_test=True)
        val_batch_generator = val_data_reader.data_generator()

    with fluid.dygraph.guard(place):
        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()

        # define data loader
        train_data_loader = fluid.io.DataLoader.from_generator(capacity=50)
        train_data_loader.set_batch_generator(batch_generator, places=place)
        if args.validation_set_dir and args.validation_set_dir != "":
            val_data_loader = fluid.io.DataLoader.from_generator(capacity=50)
            val_data_loader.set_batch_generator(val_batch_generator, places=place)

        # define model
        pretrained_bert = PretrainModelLayer(
            config=bert_config,
            return_pooled_out=True,
            weight_sharing=args.weight_sharing,
            use_fp16=args.use_fp16)

        optimizer = Optimizer(
            warmup_steps=args.warmup_steps,
            num_train_steps=args.num_train_steps,
            learning_rate=args.learning_rate,
            model_cls=pretrained_bert,
            weight_decay=args.weight_decay,
            scheduler=args.lr_scheduler,
            loss_scaling=args.loss_scaling,
            parameter_list=pretrained_bert.parameters())

        ## init from some checkpoint, to resume the previous training
        if args.init_checkpoint and args.init_checkpoint != "":
            model_dict, opt_dict = fluid.load_dygraph(
                os.path.join(args.init_checkpoint, "pretrained_bert"))
            pretrained_bert.load_dict(model_dict)
            optimizer.optimizer.set_dict(opt_dict)

        if args.use_data_parallel:
            pretrained_bert = fluid.dygraph.parallel.DataParallel(
                pretrained_bert, strategy)
            batch_generator = fluid.contrib.reader.distributed_batch_reader(
                batch_generator)

        steps = 0
        time_begin = time.time()
        time_begin_fixed = time_begin

        # train_loop
        while steps < args.num_train_steps:
            try:
                for batch in batch_generator():
                    steps += 1
                    (src_ids, pos_ids, sent_ids, input_mask, mask_label,
                        mask_pos, labels) = create_data(batch)

                    lm_acc, next_acc, mask_lm_cost, total_cost = pretrained_bert(
                        src_ids, pos_ids, sent_ids, input_mask,
                        mask_label, mask_pos, labels)

                    optimizer.optimization(total_cost, use_data_parallel=args.use_data_parallel, model=pretrained_bert)
                    pretrained_bert.clear_gradients()

                    time_end = time.time()
                    used_time = time_end - time_begin

                    epoch, current_file_index, total_file, current_file = data_reader.get_progress()
                    if steps % args.skip_steps == 0:
                        print("epoch: %d, progress: %d/%d, step: %d, loss: %f, "
                              "ppl: %f, lm_acc: %f, next_sent_acc: %f, speed: %f steps/s, file: %s"
                              % (epoch, current_file_index, total_file, steps,
                                 total_cost.numpy(),
                                 np.exp(mask_lm_cost.numpy()),
                                 lm_acc.numpy(),
                                 next_acc.numpy(), args.skip_steps / used_time,
                                 current_file))
                        time_begin = time.time()

                    if steps != 0 and steps % args.save_steps == 0 and fluid.dygraph.parallel.Env().local_rank == 0:
                        save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        fluid.save_dygraph(
                            pretrained_bert.state_dict(),
                            os.path.join(save_path, "pretrained_bert"))
                        fluid.save_dygraph(
                            optimizer.optimizer.state_dict(),
                            os.path.join(save_path, "pretrained_bert"))

                    if args.validation_set_dir and steps % args.validation_steps == 0:
                        pretrained_bert.eval()
                        vali_cost, vali_lm_cost, vali_lm_acc, vali_acc, vali_steps, vali_speed = predict_wrapper(pretrained_bert, val_data_loader)
                        print("[validation_set] epoch: %d, step: %d, "
                              "loss: %f, global ppl: %f, batch-averaged ppl: %f, "
                              "lm_acc: %f, next_sent_acc: %f, speed: %f steps/s" %
                              (epoch, steps,
                               np.mean(np.array(vali_cost) / vali_steps),
                               np.exp(np.mean(np.array(vali_lm_cost) / vali_steps)),
                               np.mean(np.exp(np.array(vali_lm_cost) / vali_steps)),
                               np.mean(np.array(vali_lm_acc) / vali_steps),
                               np.mean(np.array(vali_acc) / vali_steps), vali_speed))
                        pretrained_bert.train()

                if fluid.dygraph.parallel.Env().local_rank == 0:
                    save_path = os.path.join(args.checkpoints, "final")
                    fluid.save_dygraph(
                        pretrained_bert.state_dict(),
                        os.path.join(save_path, "pretrained_bert"))
                    fluid.save_dygraph(
                        optimizer.optimizer.state_dict(),
                        os.path.join(save_path, "pretrained_bert"))

            except fluid.core.EOFException:
                train_data_loader.reset()
                break


def test(args):
    bert_config = BertConfig(args.bert_config_path)
    bert_config.print_config()

    place = fluid.CUDAPlace(0) if args.use_cuda == True else fluid.CPUPlace()

    test_data_reader = DataReader(
        data_dir=args.test_set_dir,
        batch_size=args.batch_size,
        in_tokens=args.in_tokens,
        vocab_path=args.vocab_path,
        voc_size=bert_config['vocab_size'],
        shuffle_files=False,
        epoch=1,
        max_seq_len=args.max_seq_len,
        is_test=True)
    test_batch_generator = test_data_reader.data_generator()

    with fluid.dygraph.guard(place):
        # define data loader
        test_data_loader = fluid.io.DataLoader.from_generator(capacity=50)
        test_data_loader.set_batch_generator(test_batch_generator, places=place)

        # define model
        pretrained_bert = PretrainModelLayer(
            config=bert_config,
            return_pooled_out=True,
            weight_sharing=args.weight_sharing,
            use_fp16=args.use_fp16)

        # restore the model
        save_path = os.path.join(args.init_checkpoint, "pretrained_bert")
        print("Load params from %s" % save_path)
        model_dict, _ = fluid.load_dygraph(save_path)
        pretrained_bert.load_dict(model_dict)

        pretrained_bert.eval()

        cost, lm_cost, lm_acc, acc, steps, speed = predict_wrapper(pretrained_bert, test_data_loader)
        print("[test_set] loss: %f, global ppl: %f, lm_acc: %f, next_sent_acc: %f, speed: %f steps/s"
            % (np.mean(np.array(cost) / steps),
               np.exp(np.mean(np.array(lm_cost) / steps)),
               np.mean(np.array(lm_acc) / steps),
               np.mean(np.array(acc) / steps), speed))


if __name__ == '__main__':
    print_arguments(args)
    check_cuda(args.use_cuda)
    if args.do_test:
        test(args)
    else:
        train(args)
