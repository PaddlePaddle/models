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
""" Train and test the CoKE model on knowledge graph completion and path query datasets
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import multiprocessing
import os
import time
import logging
import json
import random

import numpy as np
import paddle
import paddle.fluid as fluid

from reader.coke_reader import KBCDataReader
from reader.coke_reader import PathqueryDataReader
from model.coke import CoKEModel
from optimization import optimization
#from evaluation import kbc_evaluation
from evaluation import kbc_batch_evaluation
from evaluation import compute_kbc_metrics
from evaluation import pathquery_batch_evaluation
from evaluation import compute_pathquery_metrics
from utils.args import ArgumentGroup, print_arguments
from utils.init import init_pretraining_params, init_checkpoint

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# yapf: disable
parser = argparse.ArgumentParser()
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("hidden_size",              int, 256,            "CoKE model config: hidden size, default 256")
model_g.add_arg("num_hidden_layers",        int, 6,              "CoKE model config: num_hidden_layers, default 6")
model_g.add_arg("num_attention_heads",      int, 4,              "CoKE model config: num_attention_heads, default 4")
model_g.add_arg("vocab_size",               int, -1,           "CoKE model config: vocab_size")
model_g.add_arg("num_relations",         int, None,           "CoKE model config: vocab_size")
model_g.add_arg("max_position_embeddings",  int, 10,             "CoKE model config: max_position_embeddings")
model_g.add_arg("hidden_act",               str, "gelu",         "CoKE model config: hidden_ac, default gelu")
model_g.add_arg("hidden_dropout_prob",      float, 0.1,          "CoKE model config: attention_probs_dropout_prob, default 0.1")
model_g.add_arg("attention_probs_dropout_prob", float, 0.1,      "CoKE model config: attention_probs_dropout_prob, default 0.1")
model_g.add_arg("initializer_range",        int, 0.02,           "CoKE model config: initializer_range")
model_g.add_arg("intermediate_size",        int, 512,            "CoKE model config: intermediate_size, default 512")

model_g.add_arg("init_checkpoint",          str,  None,          "Init checkpoint to resume training from, or for prediction only")
model_g.add_arg("init_pretraining_params",  str,  None,          "Init pre-training params which preforms fine-tuning from. If the "
                 "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
model_g.add_arg("checkpoints",              str,  "checkpoints", "Path to save checkpoints.")
model_g.add_arg("weight_sharing",           bool, True,          "If set, share weights between word embedding and masked lm.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch",             int,    100,                "Number of epoches for training.")
train_g.add_arg("learning_rate",     float,  5e-5,               "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler",     str, "linear_warmup_decay",  "scheduler of learning rate.",
                choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("soft_label",               float, 0.9,          "Value of soft labels for loss computation")
train_g.add_arg("weight_decay",      float,  0.01,               "Weight decay rate for L2 regularizer.")
train_g.add_arg("warmup_proportion", float,  0.1,                "Proportion of training steps to perform linear learning rate warmup for.")
train_g.add_arg("use_ema",           bool,   True,               "Whether to use ema.")
train_g.add_arg("ema_decay",         float,  0.9999,             "Decay rate for expoential moving average.")
train_g.add_arg("use_fp16",          bool,   False,              "Whether to use fp16 mixed precision training.")
train_g.add_arg("loss_scaling",      float,  1.0,                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps",          int,    1000,               "The steps interval to print loss.")
log_g.add_arg("verbose",             bool,   False,              "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("dataset",                   str,   "",    "dataset name")
data_g.add_arg("train_file",                str,   None,  "Data for training.")
data_g.add_arg("sen_candli_file",           str,   None,  "sentence_candicate_list file for path query evaluation. Only used for path query datasets")
data_g.add_arg("sen_trivial_file",           str,   None,  "trivial sentence file for pathquery evaluation. Only used for path query datasets")
data_g.add_arg("predict_file",              str,   None,  "Data for predictions.")
data_g.add_arg("vocab_path",                str,   None,  "Path to vocabulary.")
data_g.add_arg("true_triple_path",          str,   None,  "Path to all true triples. Only used for KBC evaluation.")
data_g.add_arg("max_seq_len",               int,   3,     "Number of tokens of the longest sequence.")
data_g.add_arg("batch_size",                int,   12,    "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("in_tokens",                 bool,  False,
               "If set, the batch size will be the maximum number of tokens in one batch. "
               "Otherwise, it will be the maximum number of examples in one batch.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("do_train",                     bool,   False,  "Whether to perform training.")
run_type_g.add_arg("do_predict",                   bool,   False,  "Whether to perform prediction.")
run_type_g.add_arg("use_cuda",                     bool,   True,   "If set, use GPU for training, default is True.")
run_type_g.add_arg("use_fast_executor",            bool,   False,  "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("num_iteration_per_drop_scope", int,    1,      "Ihe iteration intervals to clean up temporary variables.")

args = parser.parse_args()
# yapf: enable.


def create_model(pyreader_name, coke_config):
    pyreader = fluid.layers.py_reader\
            (
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, 1], [-1, 1]],
        dtypes=[
            'int64', 'int64', 'float32', 'int64', 'int64'],
        lod_levels=[0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)
    (src_ids, pos_ids, input_mask, mask_labels, mask_positions) = fluid.layers.read_file(pyreader)

    coke = CoKEModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        input_mask=input_mask,
        config=coke_config,
        soft_label=args.soft_label,
        weight_sharing=args.weight_sharing,
        use_fp16=args.use_fp16)

    loss, fc_out = coke.get_pretraining_output(mask_label=mask_labels, mask_pos=mask_positions)
    if args.use_fp16 and args.loss_scaling > 1.0:
        loss = loss * args.loss_scaling

    batch_ones = fluid.layers.fill_constant_batch_size_like(
        input=mask_labels, dtype='int64', shape=[1], value=1)
    num_seqs = fluid.layers.reduce_sum(input=batch_ones)

    return pyreader, loss, fc_out, num_seqs


def pathquery_predict(test_exe, test_program, test_pyreader, fetch_list, all_examples,
                      sen_negli_dict, trivial_sen_set, eval_result_file):
    eval_i = 0
    step = 0
    batch_mqs = []
    batch_ranks = []
    test_pyreader.start()
    while True:
        try:
            np_fc_out = test_exe.run(fetch_list=fetch_list, program=test_program)[0]
            mqs, ranks = pathquery_batch_evaluation(eval_i, all_examples, np_fc_out,
                                                    sen_negli_dict, trivial_sen_set)
            batch_mqs.extend(mqs)
            batch_ranks.extend(ranks)
            step += 1
            if step % 10 == 0:
                logger.info("Processing pathquery_predict step:%d example: %d" % (step, eval_i))
            _batch_len = np_fc_out.shape[0]
            eval_i += _batch_len
        except fluid.core.EOFException:
            test_pyreader.reset()
            break

    eval_result = compute_pathquery_metrics(batch_mqs, batch_ranks, eval_result_file)
    return eval_result


def kbc_predict(test_exe, test_program, test_pyreader, fetch_list, all_examples, true_triplets_dict, eval_result_file):
    eval_i = 0
    step = 0
    batch_eval_rets = []
    f_batch_eval_rets = []
    test_pyreader.start()
    while True:
        try:
            batch_results = []
            np_fc_out = test_exe.run(fetch_list=fetch_list, program=test_program)[0]
            _batch_len = np_fc_out.shape[0]
            for idx in range(np_fc_out.shape[0]):
                logits = [float(x) for x in np_fc_out[idx].flat]
                batch_results.append(logits)
            rank, frank = kbc_batch_evaluation(eval_i, all_examples, batch_results, true_triplets_dict)
            batch_eval_rets.extend(rank)
            f_batch_eval_rets.extend(frank)
            if step % 10 == 0:
                logger.info("Processing kbc_predict step: %d exmaples:%d" % (step, eval_i))
            step += 1
            eval_i += _batch_len
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    eval_result = compute_kbc_metrics(batch_eval_rets, f_batch_eval_rets, eval_result_file)
    return eval_result


def predict(test_exe, test_program, test_pyreader, fetch_list, all_examples, args):
    dataset = args.dataset
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    eval_result_file = os.path.join(args.checkpoints, "eval_result.json")
    logger.info(">> Evaluation result file: %s" % eval_result_file)

    if dataset.lower() in ["pathquerywn", "pathqueryfb"]:
        sen_candli_dict, trivial_sen_set = _load_pathquery_eval_dict(args.sen_candli_file,
                                                                   args.sen_trivial_file)
        logger.debug(">> Load sen_candli_dict size: %d" % len(sen_candli_dict))
        logger.debug(">> Trivial sen set size: %d" % len(trivial_sen_set))
        logger.debug(">> Finish load sen_candli set at:{}".format(time.ctime()))
        eval_performance = pathquery_predict(test_exe, test_program, test_pyreader, fetch_list,
                                              all_examples, sen_candli_dict, trivial_sen_set,
                                              eval_result_file)

        outs = "%s\t%.3f\t%.3f" % (args.dataset, eval_performance['mq'], eval_performance['fhits10'])
        logger.info("\n---------- Evaluation Performance --------------\n%s\n%s" %
                    ("\t".join(["TASK", "MQ", "Hits@10"]), outs))
    else:
        true_triplets_dict = _load_kbc_eval_dict(args.true_triple_path)
        logger.info(">> Finish loading true triplets dict %s" % time.ctime())
        eval_performance = kbc_predict(test_exe, test_program, test_pyreader, fetch_list,
                                        all_examples, true_triplets_dict, eval_result_file)
        outs = "%s\t%.3f\t%.3f\t%.3f\t%.3f" % (args.dataset,
                                               eval_performance['fmrr'],
                                               eval_performance['fhits1'],
                                               eval_performance['fhits3'],
                                               eval_performance['fhits10'])
        logger.info("\n----------- Evaluation Performance --------------\n%s\n%s" %
                    ("\t".join(["TASK", "MRR", "Hits@1", "Hits@3", "Hits@10"]), outs))
    return eval_performance


def _load_kbc_eval_dict(true_triple_file):
    def load_true_triples(true_triple_file):
        true_triples = []
        with open(true_triple_file, "r") as fr:
            for line in fr.readlines():
                tokens = line.strip("\r \n").split("\t")
                assert len(tokens) == 3
                true_triples.append(
                    (int(tokens[0]), int(tokens[1]), int(tokens[2])))
        logger.debug("Finish loading %d true triples" % len(true_triples))
        return true_triples
    true_triples = load_true_triples(true_triple_file)
    true_triples_dict = collections.defaultdict(lambda: {'hs': collections.defaultdict(list),
                                          'ts': collections.defaultdict(list)})
    for h, r, t in true_triples:
        true_triples_dict[r]['ts'][h].append(t)
        true_triples_dict[r]['hs'][t].append(h)
    return true_triples_dict


def _load_pathquery_eval_dict(sen_candli_file, trivial_sen_file, add_gold_o = True):
    sen_candli_dict = dict()
    for line in open(sen_candli_file):
        line = line.strip()
        segs = line.split("\t")
        assert len(segs) == 2, " Illegal format for sen_candli_dict, expects 2 columns data"
        sen = segs[0]
        candset = set(segs[1].split(" "))
        if add_gold_o is True:
            gold_o = sen.split(" ")[-1]
            candset.add(gold_o)
        _li = list(candset)
        int_li = [int(x) for x in _li]
        sen_candli_dict[sen] = int_li
    trivial_senset = {x.strip() for x in open(trivial_sen_file)}

    return sen_candli_dict, trivial_senset


def init_coke_net_config(args, print_config = True):
    config = dict()
    config["hidden_size"] = args.hidden_size
    config["num_hidden_layers"] = args.num_hidden_layers
    config["num_attention_heads"] = args.num_attention_heads
    config["vocab_size"] = args.vocab_size
    config["num_relations"] = args.num_relations
    config["max_position_embeddings"] = args.max_position_embeddings
    config["hidden_act"] = args.hidden_act
    config["hidden_dropout_prob"] = args.hidden_dropout_prob
    config["attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
    config["initializer_range"] = args.initializer_range
    config["intermediate_size"] = args.intermediate_size

    if print_config is True:
        logger.info('----------- CoKE Network Configuration -------------')
        for arg, value in config.items():
            logger.info('%s: %s' % (arg, value))
        logger.info('------------------------------------------------')
    return config


def main(args):
    if not (args.do_train or args.do_predict):
        raise ValueError("For args `do_train` and `do_predict`, at "
                         "least one of them must be True.")
    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    startup_prog = fluid.Program()

    # Init programs
    coke_config = init_coke_net_config(args, print_config=True)
    if args.do_train:
        train_data_reader = get_data_reader(args, args.train_file, is_training=True,
                                            epoch=args.epoch, shuffle=True, dev_count=dev_count,
                                            vocab_size=args.vocab_size)

        num_train_examples = train_data_reader.total_instance
        if args.in_tokens:
            max_train_steps = args.epoch * num_train_examples // (
                    args.batch_size // args.max_seq_len) // dev_count
        else:
            max_train_steps = args.epoch * num_train_examples // (
                args.batch_size) // dev_count
        warmup_steps = int(max_train_steps * args.warmup_proportion)
        logger.info("Device count: %d" % dev_count)
        logger.info("Num train examples: %d" % num_train_examples)
        logger.info("Max train steps: %d" % max_train_steps)
        logger.info("Num warmup steps: %d" % warmup_steps)

        train_program = fluid.Program()

        # Create model and set optimization for train
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, loss, _, num_seqs = create_model(
                    pyreader_name='train_reader',
                    coke_config=coke_config)

                scheduled_lr = optimization(
                    loss=loss,
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=args.weight_decay,
                    scheduler=args.lr_scheduler,
                    use_fp16=args.use_fp16,
                    loss_scaling=args.loss_scaling)

                if args.use_ema:
                    ema = fluid.optimizer.ExponentialMovingAverage(args.ema_decay)
                    ema.update()

                fluid.memory_optimize(train_program, skip_opt_set=[loss.name, num_seqs.name])

        if args.verbose:
            if args.in_tokens:
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program,
                    batch_size=args.batch_size // args.max_seq_len)
            else:
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program, batch_size=args.batch_size)
            logger.info("Theoretical memory usage in training:  %.3f - %.3f %s" %
                        (lower_mem, upper_mem, unit))

    if args.do_predict:
        # Create model for prediction
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, _, fc_out, num_seqs = create_model(
                    pyreader_name='test_reader',
                    coke_config=coke_config)

                if args.use_ema and 'ema' not in dir():
                    ema = fluid.optimizer.ExponentialMovingAverage(args.ema_decay)

                fluid.memory_optimize(test_prog, skip_opt_set=[fc_out.name, num_seqs.name])

        test_prog = test_prog.clone(for_test=True)

    exe.run(startup_prog)

    # Init checkpoints
    if args.do_train:
        init_train_checkpoint(args, exe, startup_prog)
    elif args.do_predict:
        init_predict_checkpoint(args, exe, startup_prog)

    # Run training
    if args.do_train:
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = args.use_fast_executor
        exec_strategy.num_threads = dev_count
        exec_strategy.num_iteration_per_drop_scope = args.num_iteration_per_drop_scope

        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=loss.name,
            exec_strategy=exec_strategy,
            main_program=train_program)

        train_pyreader.decorate_tensor_provider(train_data_reader.data_generator())

        train_pyreader.start()
        steps = 0
        total_cost, total_num_seqs = [], []
        time_begin = time.time()
        while steps < max_train_steps:
            try:
                steps += 1
                if steps % args.skip_steps == 0:
                    if warmup_steps <= 0:
                        fetch_list = [loss.name, num_seqs.name]
                    else:
                        fetch_list = [
                            loss.name, scheduled_lr.name, num_seqs.name
                        ]
                else:
                    fetch_list = []

                outputs = train_exe.run(fetch_list=fetch_list)

                if steps % args.skip_steps == 0:
                    if warmup_steps <= 0:
                        np_loss, np_num_seqs = outputs
                    else:
                        np_loss, np_lr, np_num_seqs = outputs
                    total_cost.extend(np_loss * np_num_seqs)
                    total_num_seqs.extend(np_num_seqs)

                    if args.verbose:
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size(
                        )
                        verbose += "learning rate: %f" % (
                            np_lr[0]
                            if warmup_steps > 0 else args.learning_rate)
                        logger.info(verbose)

                    time_end = time.time()
                    used_time = time_end - time_begin
                    current_example, epoch = train_data_reader.get_progress()

                    logger.info("epoch: %d, progress: %d/%d, step: %d, loss: %f, "
                                "speed: %f steps/s" %
                                (epoch, current_example, num_train_examples, steps,
                                 np.sum(total_cost) / np.sum(total_num_seqs),
                                 args.skip_steps / used_time))
                    total_cost, total_num_seqs = [], []
                    time_begin = time.time()

                if steps == max_train_steps:
                    save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)
            except fluid.core.EOFException:
                logger.warning(">> EOFException")
                save_path = os.path.join(args.checkpoints, "step_" + str(steps) + "_final")
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break
        logger.info(">>Finish training at %s " % time.ctime())

    # Run prediction
    if args.do_predict:
        assert dev_count == 1, "During prediction, dev_count expects 1, current is %d" % dev_count
        test_data_reader = get_data_reader(args, args.predict_file, is_training=False,
                                           epoch=1, shuffle=False, dev_count=dev_count,
                                           vocab_size=args.vocab_size)
        test_pyreader.decorate_tensor_provider(test_data_reader.data_generator())

        if args.use_ema:
            with ema.apply(exe):
                eval_performance = predict(exe, test_prog, test_pyreader,
                                           [fc_out.name], test_data_reader.examples, args)
        else:
            eval_performance = predict(exe, test_prog, test_pyreader,
                                       [fc_out.name], test_data_reader.examples, args)

        logger.info(">>Finish predicting at %s " % time.ctime())


def init_predict_checkpoint(args, exe, startup_prog):
    if args.dataset in ["pathQueryWN", "pathQueryFB"]:
        assert args.sen_candli_file is not None and args.sen_trivial_file is not None, "during test, pathQuery sen_candli_file and path_trivial_file must be set "
    if not args.init_checkpoint:
        raise ValueError("args 'init_checkpoint' should be set if"
                         "only doing prediction!")
    init_checkpoint(
        exe,
        args.init_checkpoint,
        main_program=startup_prog,
        use_fp16=args.use_fp16)


def init_train_checkpoint(args, exe, startup_prog):
    if args.init_checkpoint and args.init_pretraining_params:
        logger.info(
            "WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
            "both are set! Only arg 'init_checkpoint' is made valid.")
    if args.init_checkpoint:
        init_checkpoint(
            exe,
            args.init_checkpoint,
            main_program=startup_prog,
            use_fp16=args.use_fp16,
            print_var_verbose=False)
    elif args.init_pretraining_params:
        init_pretraining_params(
            exe,
            args.init_pretraining_params,
            main_program=startup_prog,
            use_fp16=args.use_fp16)


def get_data_reader(args, data_file, epoch, is_training, shuffle, dev_count, vocab_size):
    if args.dataset.lower() in ["pathqueryfb", "pathquerywn"]:
        Reader = PathqueryDataReader
    else:
        Reader = KBCDataReader
    data_reader = Reader(
        vocab_path=args.vocab_path,
        data_path=data_file,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        is_training=is_training,
        shuffle=shuffle,
        dev_count=dev_count,
        epoch=epoch,
        vocab_size=vocab_size)
    return data_reader


if __name__ == '__main__':
    print_arguments(args)
    main(args)
