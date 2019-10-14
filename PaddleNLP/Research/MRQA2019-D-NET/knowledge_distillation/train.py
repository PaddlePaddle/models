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

import os
import time
import argparse
import collections
import numpy as np
import multiprocessing

import paddle
import paddle.fluid as fluid

from utils.placeholder import Placeholder
from utils.init import init_pretraining_params, init_checkpoint
from utils.configure import ArgumentGroup, print_arguments, JsonConfig

from model import mlm_net
from model import mrqa_net

from optimizer.optimization import optimization
from model.bert_model import ModelBERT
from reader.mrqa_reader import DataProcessor, write_predictions
from reader.mrqa_distill_reader import DataProcessorDistill 
from reader.mlm_reader import DataReader
from reader.joint_reader import create_reader


parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("bert_config_path", str, None, "Path to the json file for bert model config.")
model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
model_g.add_arg("init_pretraining_params", str, None,
                "Init pre-training params which preforms fine-tuning from. If the "
                "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
model_g.add_arg("checkpoints", str, "checkpoints", "Path to save checkpoints.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 3, "Number of epoches for fine-tuning.")
train_g.add_arg("learning_rate", float, 5e-5, "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler", str, "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay", float, 0.01, "Weight decay rate for L2 regularizer.")
train_g.add_arg("use_ema", bool, True, "Whether to use ema.")
train_g.add_arg("ema_decay", float, 0.9999, "Decay rate for expoential moving average.")
train_g.add_arg("warmup_proportion", float, 0.1,
                "Proportion of training steps to perform linear learning rate warmup for.")
train_g.add_arg("save_steps", int, 1000, "The steps interval to save checkpoints.")
train_g.add_arg("sample_rate", float, 0.02, "train samples num.")
train_g.add_arg("use_fp16", bool, False, "Whether to use fp16 mixed precision training.")
train_g.add_arg("mix_ratio", float, 0.4, "batch mix ratio for masked language model task")
train_g.add_arg("loss_scaling", float, 1.0,
                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")

train_g.add_arg("do_distill", bool, False, "do distillation")

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")
log_g.add_arg("verbose", bool, False, "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("train_file", str, None, "json data for training.")
data_g.add_arg("mlm_path", str, None, "data for masked language model training.")
data_g.add_arg("predict_file", str, None, "json data for predictions.")
data_g.add_arg("vocab_path", str, None, "Vocabulary path.")
data_g.add_arg("with_negative", bool, False,
               "If true, the examples contain some that do not have an answer.")
data_g.add_arg("max_seq_len", int, 512, "Number of words of the longest seqence.")
data_g.add_arg("max_query_length", int, 64, "Max query length.")
data_g.add_arg("max_answer_length", int, 30, "Max answer length.")
data_g.add_arg("batch_size", int, 12,
               "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("in_tokens", bool, False,
               "If set, the batch size will be the maximum number of tokens in one batch. "
               "Otherwise, it will be the maximum number of examples in one batch.")
data_g.add_arg("do_lower_case", bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
data_g.add_arg("doc_stride", int, 128,
               "When splitting up a long document into chunks, how much stride to take between chunks.")
data_g.add_arg("n_best_size", int, 20,
               "The total number of n-best predictions to generate in the nbest_predictions.json output file.")
data_g.add_arg("null_score_diff_threshold", float, 0.0,
               "If null_score - best_non_null is greater than the threshold predict null.")
data_g.add_arg("random_seed", int, 0, "Random seed.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
run_type_g.add_arg("use_fast_executor", bool, False,
                   "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("num_iteration_per_drop_scope", int, 1,
                   "Ihe iteration intervals to clean up temporary variables.")
run_type_g.add_arg("do_train", bool, True, "Whether to perform training.")
run_type_g.add_arg("do_predict", bool, True, "Whether to perform prediction.")

args = parser.parse_args()


max_seq_len = args.max_seq_len

if args.do_distill: 
    input_shape = [
        ([1, 1], 'int64'),
        ([-1, max_seq_len, 1], 'int64'), # src_ids
        ([-1, max_seq_len, 1], 'int64'), # pos_ids
        ([-1, max_seq_len, 1], 'int64'), # sent_ids
        ([-1, max_seq_len, 1], 'float32'), # input_mask
        ([-1, max_seq_len, 1], 'float32'), # start_logits_truth
        ([-1, max_seq_len, 1], 'float32'), # end_logits_truth
        ([-1, 1], 'int64'),  # start label
        ([-1, 1], 'int64'),  # end label
        ([-1, 1], 'int64'),  # masked label
        ([-1, 1], 'int64')]  # masked pos
else: 
    input_shape = [
        ([1, 1], 'int64'),
        ([-1, max_seq_len, 1], 'int64'),
        ([-1, max_seq_len, 1], 'int64'),
        ([-1, max_seq_len, 1], 'int64'),
        ([-1, max_seq_len, 1], 'float32'),
        ([-1, 1], 'int64'),  # start label
        ([-1, 1], 'int64'),  # end label
        ([-1, 1], 'int64'),  # masked label
        ([-1, 1], 'int64')]  # masked pos

# yapf: enable.

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def predict(test_exe, test_program, test_pyreader, fetch_list, processor, prefix=''):
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    output_prediction_file = os.path.join(args.checkpoints, prefix + "predictions.json")
    output_nbest_file = os.path.join(args.checkpoints, prefix + "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(args.checkpoints, prefix + "null_odds.json")

    test_pyreader.start()
    all_results = []
    time_begin = time.time()
    while True:
        try:
            np_unique_ids, np_start_logits, np_end_logits, np_num_seqs = test_exe.run(
                fetch_list=fetch_list, program=test_program)
            for idx in range(np_unique_ids.shape[0]):
                if np_unique_ids[idx] < 0:
                    continue
                if len(all_results) % 1000 == 0:
                    print("Processing example: %d" % len(all_results))
                unique_id = int(np_unique_ids[idx])
                start_logits = [float(x) for x in np_start_logits[idx].flat]
                end_logits = [float(x) for x in np_end_logits[idx].flat]
                all_results.append(
                    RawResult(
                        unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits))
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()

    features = processor.get_features(
        processor.predict_examples, is_training=False)
    write_predictions(processor.predict_examples, features, all_results,
                      args.n_best_size, args.max_answer_length,
                      args.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file,
                      args.with_negative,
                      args.null_score_diff_threshold, args.verbose)


def train(args):

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

    startup_prog = fluid.default_startup_program()

    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.do_train: 
        if args.do_distill: 
            train_processor = DataProcessorDistill()
            mrc_train_generator = train_processor.data_generator(
                data_file=args.train_file,
                batch_size=args.batch_size,
                max_len=args.max_seq_len,
                in_tokens=False,
                dev_count=dev_count,
                epochs=args.epoch,
                shuffle=True)
        else: 
            train_processor = DataProcessor(
                vocab_path=args.vocab_path,
                do_lower_case=args.do_lower_case,
                max_seq_length=args.max_seq_len,
                in_tokens=args.in_tokens,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length)

            mrc_train_generator = train_processor.data_generator(
                data_path=args.train_file,
                batch_size=args.batch_size,
                max_len=args.max_seq_len,
                phase='train',
                shuffle=True,
                dev_count=dev_count,
                with_negative=args.with_negative,
                epoch=args.epoch)

        bert_conf = JsonConfig(args.bert_config_path)
        
        data_reader = DataReader(
            args.mlm_path,
            vocab_path=args.vocab_path,
            batch_size=args.batch_size,
            in_tokens=args.in_tokens,
            voc_size=bert_conf['vocab_size'],
            shuffle_files=False,
            epoch=args.epoch,
            max_seq_len=args.max_seq_len,
            is_test=False)
        mlm_train_generator = data_reader.data_generator()
        gens = [
            (mrc_train_generator, 1.0),
            (mlm_train_generator, args.mix_ratio)
        ]
        # create joint pyreader
        joint_generator, train_pyreader, model_inputs = \
            create_reader("train_reader", input_shape, True, args.do_distill, 
                          gens)
        train_pyreader.decorate_tensor_provider(joint_generator)

        task_id = model_inputs[0]
        if args.do_distill: 
            bert_inputs = model_inputs[1:5]
            mrc_inputs = model_inputs[1:9]
            mlm_inputs = model_inputs[9:11]
        else: 
            bert_inputs = model_inputs[1:5]
            mrc_inputs = model_inputs[1:7]
            mlm_inputs = model_inputs[7:9]
        
        # create model
        train_bert_model = ModelBERT(
            conf={"bert_conf_file": args.bert_config_path},
            is_training=True)
        train_create_bert = train_bert_model.create_model(args, bert_inputs)

        build_strategy = fluid.BuildStrategy()
        if args.do_distill: 
            num_train_examples = train_processor.num_examples
            print("runtime number of examples:")
            print(num_train_examples)
        else: 
            print("estimating runtime number of examples...")
            num_train_examples = train_processor.estimate_runtime_examples(
                args.train_file, sample_rate=args.sample_rate)
            print("runtime number of examples:")
            print(num_train_examples)

        if args.in_tokens:
            max_train_steps = args.epoch * num_train_examples // (
                    args.batch_size // args.max_seq_len) // dev_count
        else:
            max_train_steps = args.epoch * num_train_examples // (
                args.batch_size) // dev_count
        max_train_steps = int(max_train_steps * (1 + args.mix_ratio))
        warmup_steps = int(max_train_steps * args.warmup_proportion)
        print("Device count: %d" % dev_count)
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)
        print("Num warmup steps: %d" % warmup_steps)

        train_program = fluid.default_main_program()
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_create_bert()
                mlm_output_tensors = mlm_net.create_model(
                    mlm_inputs, base_model=train_bert_model, is_training=True, args=args
                )
                mrc_output_tensors = mrqa_net.create_model(
                    mrc_inputs, base_model=train_bert_model, is_training=True, args=args
                )
                task_one_hot = fluid.layers.one_hot(task_id, 2)
                mrc_loss = mrqa_net.compute_loss(mrc_output_tensors, args)
                if args.do_distill: 
                    distill_loss = mrqa_net.compute_distill_loss(mrc_output_tensors, args)
                    mrc_loss = mrc_loss + distill_loss
                num_seqs = mrc_output_tensors['num_seqs']
                mlm_loss = mlm_net.compute_loss(mlm_output_tensors)
                num_seqs = mlm_output_tensors['num_seqs']
                all_loss = fluid.layers.concat([mrc_loss, mlm_loss], axis=0)
                loss = fluid.layers.reduce_sum(task_one_hot * all_loss)
                
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

                loss.persistable = True
                num_seqs.persistable = True

                ema = fluid.optimizer.ExponentialMovingAverage(args.ema_decay)
                ema.update()

        train_compiled_program = fluid.CompiledProgram(train_program).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy)

        if args.verbose:
            if args.in_tokens:
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program,
                    batch_size=args.batch_size // args.max_seq_len)
            else:
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program, batch_size=args.batch_size)
            print("Theoretical memory usage in training:  %.3f - %.3f %s" %
                  (lower_mem, upper_mem, unit))

    if args.do_predict:
        predict_processor = DataProcessor(
            vocab_path=args.vocab_path,
            do_lower_case=args.do_lower_case,
            max_seq_length=args.max_seq_len,
            in_tokens=args.in_tokens,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length)
        mrc_test_generator = predict_processor.data_generator(
            data_path=args.predict_file,
            batch_size=args.batch_size,
            max_len=args.max_seq_len,
            phase='predict',
            shuffle=False,
            dev_count=dev_count,
            epoch=1)

        test_input_shape = [
            ([-1, max_seq_len, 1], 'int64'),
            ([-1, max_seq_len, 1], 'int64'),
            ([-1, max_seq_len, 1], 'int64'),
            ([-1, max_seq_len, 1], 'float32'),
            ([-1, 1], 'int64')]
        build_strategy = fluid.BuildStrategy()
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                placeholder = Placeholder(test_input_shape)
                test_pyreader, model_inputs = placeholder.build(
                    capacity=100, reader_name="test_reader")

                test_pyreader.decorate_tensor_provider(mrc_test_generator)

                # create model
                bert_inputs = model_inputs[0:4]
                mrc_inputs = model_inputs
                test_bert_model = ModelBERT(
                    conf={"bert_conf_file": args.bert_config_path},
                    is_training=False)
                test_create_bert = test_bert_model.create_model(args, bert_inputs)

                test_create_bert()
                mrc_output_tensors = mrqa_net.create_model(
                    mrc_inputs, base_model=test_bert_model, is_training=False, args=args
                )
                unique_ids = mrc_output_tensors['unique_id']
                start_logits = mrc_output_tensors['start_logits']
                end_logits = mrc_output_tensors['end_logits']
                num_seqs = mrc_output_tensors['num_seqs']

                if 'ema' not in dir():
                    ema = fluid.optimizer.ExponentialMovingAverage(args.ema_decay)

                unique_ids.persistable = True
                start_logits.persistable = True
                end_logits.persistable = True
                num_seqs.persistable = True

        test_prog = test_prog.clone(for_test=True)
        test_compiled_program = fluid.CompiledProgram(test_prog).with_data_parallel(
            build_strategy=build_strategy)

    exe.run(startup_prog)

    if args.do_train:
        if args.init_checkpoint and args.init_pretraining_params:
            print(
                "WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
                "both are set! Only arg 'init_checkpoint' is made valid.")
        if args.init_checkpoint:
            init_checkpoint(
                exe,
                args.init_checkpoint,
                main_program=startup_prog,
                use_fp16=args.use_fp16)
        elif args.init_pretraining_params:
            init_pretraining_params(
                exe,
                args.init_pretraining_params,
                main_program=startup_prog,
                use_fp16=args.use_fp16)
    elif args.do_predict:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing prediction!")
        init_checkpoint(
            exe,
            args.init_checkpoint,
            main_program=startup_prog,
            use_fp16=args.use_fp16)

    if args.do_train:
        train_pyreader.start()

        steps = 0
        total_cost, total_num_seqs = [], []
        time_begin = time.time()
        while True:
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

                outputs = exe.run(train_compiled_program, fetch_list=fetch_list)

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
                        print(verbose)

                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("progress: %d/%d, step: %d, loss: %f" % (steps, max_train_steps, steps, np.sum(total_cost) / np.sum(total_num_seqs)))
                    
                    total_cost, total_num_seqs = [], []
                    time_begin = time.time()

                if steps % args.save_steps == 0:
                    save_path = os.path.join(args.checkpoints,
                                             "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)
                if steps == max_train_steps:
                    save_path = os.path.join(args.checkpoints,
                                             "step_" + str(steps) + "_final")
                    fluid.io.save_persistables(exe, save_path, train_program)
                    break
            except paddle.fluid.core.EOFException as err:
                save_path = os.path.join(args.checkpoints,
                                         "step_" + str(steps) + "_final")
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break

    if args.do_predict:
        if args.use_ema:
            with ema.apply(exe):
                predict(exe, test_compiled_program, test_pyreader, [
                    unique_ids.name, start_logits.name, end_logits.name, num_seqs.name
                ], predict_processor, prefix='ema_')
        else:
            predict(exe, test_compiled_program, test_pyreader, [
                unique_ids.name, start_logits.name, end_logits.name, num_seqs.name
            ], predict_processor)


if __name__ == '__main__':
    print_arguments(args)
    train(args)
