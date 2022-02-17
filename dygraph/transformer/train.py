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

import logging
import os
import six
import sys
import time

import numpy as np
import paddle
import paddle.fluid as fluid

from utils.configure import PDConfig
from utils.check import check_gpu, check_version

# include task-specific libs
import reader
from model import Transformer, CrossEntropyCriterion, NoamDecay

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def do_train(args):
    if args.use_cuda:
        trainer_count = fluid.dygraph.parallel.Env().nranks
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env(
        ).dev_id) if trainer_count > 1 else fluid.CUDAPlace(0)
    else:
        trainer_count = 1
        place = fluid.CPUPlace()

    # define the data generator
    processor = reader.DataProcessor(
        fpattern=args.training_file,
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        token_delimiter=args.token_delimiter,
        use_token_batch=args.use_token_batch,
        batch_size=args.batch_size,
        device_count=trainer_count,
        pool_size=args.pool_size,
        sort_type=args.sort_type,
        shuffle=args.shuffle,
        shuffle_batch=args.shuffle_batch,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        max_length=args.max_length,
        n_head=args.n_head)
    batch_generator = processor.data_generator(phase="train")
    if args.validation_file:
        val_processor = reader.DataProcessor(
            fpattern=args.validation_file,
            src_vocab_fpath=args.src_vocab_fpath,
            trg_vocab_fpath=args.trg_vocab_fpath,
            token_delimiter=args.token_delimiter,
            use_token_batch=args.use_token_batch,
            batch_size=args.batch_size,
            device_count=trainer_count,
            pool_size=args.pool_size,
            sort_type=args.sort_type,
            shuffle=False,
            shuffle_batch=False,
            start_mark=args.special_token[0],
            end_mark=args.special_token[1],
            unk_mark=args.special_token[2],
            max_length=args.max_length,
            n_head=args.n_head)
        val_batch_generator = val_processor.data_generator(phase="train")
    if trainer_count > 1:  # for multi-process gpu training
        batch_generator = fluid.contrib.reader.distributed_batch_reader(
            batch_generator)
    args.src_vocab_size, args.trg_vocab_size, args.bos_idx, args.eos_idx, \
        args.unk_idx = processor.get_vocab_summary()

    with fluid.dygraph.guard(place):
        # set seed for CE
        random_seed = eval(str(args.random_seed))
        if random_seed is not None:
            fluid.default_main_program().random_seed = random_seed
            fluid.default_startup_program().random_seed = random_seed

        # define data loader
        train_loader = fluid.io.DataLoader.from_generator(capacity=10)
        train_loader.set_batch_generator(batch_generator, places=place)
        if args.validation_file:
            val_loader = fluid.io.DataLoader.from_generator(capacity=10)
            val_loader.set_batch_generator(val_batch_generator, places=place)

        # define model
        transformer = Transformer(
            args.src_vocab_size, args.trg_vocab_size, args.max_length + 1,
            args.n_layer, args.n_head, args.d_key, args.d_value, args.d_model,
            args.d_inner_hid, args.prepostprocess_dropout,
            args.attention_dropout, args.relu_dropout, args.preprocess_cmd,
            args.postprocess_cmd, args.weight_sharing, args.bos_idx,
            args.eos_idx)

        # define loss
        criterion = CrossEntropyCriterion(args.label_smooth_eps)

        # define optimizer
        optimizer = fluid.optimizer.Adam(
            learning_rate=NoamDecay(args.d_model, args.warmup_steps,
                                    args.learning_rate),
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=float(args.eps),
            parameter_list=transformer.parameters())

        ## init from some checkpoint, to resume the previous training
        if args.init_from_checkpoint:
            model_dict, opt_dict = fluid.load_dygraph(
                os.path.join(args.init_from_checkpoint, "transformer"))
            transformer.load_dict(model_dict)
            optimizer.set_dict(opt_dict)
        ## init from some pretrain models, to better solve the current task
        if args.init_from_pretrain_model:
            model_dict, _ = fluid.load_dygraph(
                os.path.join(args.init_from_pretrain_model, "transformer"))
            transformer.load_dict(model_dict)

        if trainer_count > 1:
            strategy = fluid.dygraph.parallel.prepare_context()
            transformer = fluid.dygraph.parallel.DataParallel(transformer,
                                                              strategy)

        # the best cross-entropy value with label smoothing
        loss_normalizer = -(
            (1. - args.label_smooth_eps) * np.log(
                (1. - args.label_smooth_eps)) + args.label_smooth_eps *
            np.log(args.label_smooth_eps / (args.trg_vocab_size - 1) + 1e-20))

        ce_time = []
        ce_ppl = []
        step_idx = 0

        # train loop
        for pass_id in range(args.epoch):
            epoch_start = time.time()

            batch_id = 0
            batch_start = time.time()
            interval_word_num = 0.0
            for input_data in train_loader():
                if args.max_iter and step_idx == args.max_iter:  #NOTE: used for benchmark
                    return
                batch_reader_end = time.time()

                (src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
                 trg_slf_attn_bias, trg_src_attn_bias, lbl_word,
                 lbl_weight) = input_data

                logits = transformer(src_word, src_pos, src_slf_attn_bias,
                                     trg_word, trg_pos, trg_slf_attn_bias,
                                     trg_src_attn_bias)

                sum_cost, avg_cost, token_num = criterion(logits, lbl_word,
                                                          lbl_weight)

                # NOTE: When using PaddlePaddle 2.0, it's not necessary to call
                # scale_loss() and apply_collective_grads(). However, they are both
                # necessary for PaddlePaddle 1.8. Please check PaddlePaddle version. 
                avg_cost.backward()

                optimizer.minimize(avg_cost)
                transformer.clear_gradients()

                interval_word_num += np.prod(src_word.shape)
                if step_idx % args.print_step == 0:
                    total_avg_cost = avg_cost.numpy()

                    if step_idx == 0:
                        logger.info(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f" %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)])))
                    else:
                        train_avg_batch_cost = args.print_step / (
                            time.time() - batch_start)
                        word_speed = interval_word_num / (
                            time.time() - batch_start)
                        logger.info(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f, avg_speed: %.2f step/s, "
                            "words speed: %0.2f words/s" %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)]),
                             train_avg_batch_cost, word_speed))
                    batch_start = time.time()
                    interval_word_num = 0.0

                if step_idx % args.save_step == 0 and step_idx != 0:
                    # validation
                    if args.validation_file:
                        transformer.eval()
                        total_sum_cost = 0
                        total_token_num = 0
                        for input_data in val_loader():
                            (src_word, src_pos, src_slf_attn_bias, trg_word,
                             trg_pos, trg_slf_attn_bias, trg_src_attn_bias,
                             lbl_word, lbl_weight) = input_data
                            logits = transformer(
                                src_word, src_pos, src_slf_attn_bias, trg_word,
                                trg_pos, trg_slf_attn_bias, trg_src_attn_bias)
                            sum_cost, avg_cost, token_num = criterion(
                                logits, lbl_word, lbl_weight)
                            total_sum_cost += sum_cost.numpy()
                            total_token_num += token_num.numpy()
                            total_avg_cost = total_sum_cost / total_token_num
                        logger.info("validation, step_idx: %d, avg loss: %f, "
                                    "normalized loss: %f, ppl: %f" %
                                    (step_idx, total_avg_cost,
                                     total_avg_cost - loss_normalizer,
                                     np.exp([min(total_avg_cost, 100)])))
                        transformer.train()

                    if args.save_model and (
                            trainer_count == 1 or
                            fluid.dygraph.parallel.Env().dev_id == 0):
                        model_dir = os.path.join(args.save_model,
                                                 "step_" + str(step_idx))
                        if not os.path.exists(model_dir):
                            os.makedirs(model_dir)
                        fluid.save_dygraph(
                            transformer.state_dict(),
                            os.path.join(model_dir, "transformer"))
                        fluid.save_dygraph(
                            optimizer.state_dict(),
                            os.path.join(model_dir, "transformer"))

                batch_id += 1
                step_idx += 1

            train_epoch_cost = time.time() - epoch_start
            ce_time.append(train_epoch_cost)
            logger.info("train epoch: %d, epoch_cost: %.5f s" %
                        (pass_id, train_epoch_cost))

        if args.save_model:
            model_dir = os.path.join(args.save_model, "step_final")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            fluid.save_dygraph(transformer.state_dict(),
                               os.path.join(model_dir, "transformer"))
            fluid.save_dygraph(optimizer.state_dict(),
                               os.path.join(model_dir, "transformer"))

        if args.enable_ce:
            _ppl = 0
            _time = 0
            try:
                _time = ce_time[-1]
                _ppl = ce_ppl[-1]
            except:
                print("ce info error")
            print("kpis\ttrain_duration_card%s\t%s" % (trainer_count, _time))
            print("kpis\ttrain_ppl_card%s\t%f" % (trainer_count, _ppl))


if __name__ == "__main__":
    args = PDConfig(yaml_file="./transformer.yaml")
    args.build()
    args.Print()
    check_gpu(args.use_cuda)
    check_version()

    do_train(args)
