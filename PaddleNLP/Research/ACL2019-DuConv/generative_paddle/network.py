#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: network.py
"""

import argparse

import numpy as np

import paddle.fluid as fluid
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor

from source.utils.utils import str2bool
from source.utils.utils import id_to_text
from source.utils.utils import init_embedding
from source.utils.utils import build_data_feed
from source.utils.utils import load_id2str_dict
from source.inputters.corpus import KnowledgeCorpus
from source.models.knowledge_seq2seq import knowledge_seq2seq


def model_config():
    """ model config """
    parser = argparse.ArgumentParser()

    # Data
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--data_dir", type=str, default="./data/")
    data_arg.add_argument("--data_prefix", type=str, default="demo")
    data_arg.add_argument("--save_dir", type=str, default="./models/")
    data_arg.add_argument("--vocab_path", type=str, default="./data/vocab.txt")
    data_arg.add_argument("--embed_file", type=str,
                          default="./data/sgns.weibo.300d.txt")

    # Network
    net_arg = parser.add_argument_group("Network")
    net_arg.add_argument("--embed_size", type=int, default=300)
    net_arg.add_argument("--hidden_size", type=int, default=800)
    net_arg.add_argument("--bidirectional", type=str2bool, default=True)
    net_arg.add_argument("--vocab_size", type=int, default=30004)
    net_arg.add_argument("--min_len", type=int, default=1)
    net_arg.add_argument("--max_len", type=int, default=500)
    net_arg.add_argument("--num_layers", type=int, default=1)
    net_arg.add_argument("--attn", type=str, default='dot',
                         choices=['none', 'mlp', 'dot', 'general'])

    # Training / Testing
    train_arg = parser.add_argument_group("Training")

    train_arg.add_argument("--stage", type=int, default="0")
    train_arg.add_argument("--run_type", type=str, default="train")
    train_arg.add_argument("--init_model", type=str, default="")
    train_arg.add_argument("--init_opt_state", type=str, default="")

    train_arg.add_argument("--optimizer", type=str, default="Adam")
    train_arg.add_argument("--lr", type=float, default=0.0005)
    train_arg.add_argument("--grad_clip", type=float, default=5.0)
    train_arg.add_argument("--dropout", type=float, default=0.3)
    train_arg.add_argument("--num_epochs", type=int, default=13)
    train_arg.add_argument("--pretrain_epoch", type=int, default=5)
    train_arg.add_argument("--use_bow", type=str2bool, default=True)
    train_arg.add_argument("--use_posterior", type=str2bool, default=False)

    # Geneation
    gen_arg = parser.add_argument_group("Generation")
    gen_arg.add_argument("--beam_size", type=int, default=10)
    gen_arg.add_argument("--max_dec_len", type=int, default=30)
    gen_arg.add_argument("--length_average", type=str2bool, default=True)
    gen_arg.add_argument("--output", type=str, default="./output/test.result")
    gen_arg.add_argument("--model_path", type=str, default="./models/best_model/")
    gen_arg.add_argument("--unk_id", type=int, default=1)
    gen_arg.add_argument("--bos_id", type=int, default=2)
    gen_arg.add_argument("--eos_id", type=int, default=3)

    # MISC
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument("--use_gpu", type=str2bool, default=True)
    misc_arg.add_argument("--log_steps", type=int, default=300)
    misc_arg.add_argument("--valid_steps", type=int, default=1000)
    misc_arg.add_argument("--batch_size", type=int, default=1)

    config = parser.parse_args()

    return config


def trace_fianl_result(final_score, final_ids, final_index, topk=1, EOS=3):
    """ trace fianl result """
    col_size = final_score.shape[1]
    row_size = final_score.shape[0]

    found_eos_num = 0
    i = row_size - 1

    beam_size = col_size
    score = final_score[-1]
    row_array = [row_size - 1] * beam_size
    col_array = [e for e in range(col_size)]

    while i >= 0:
        for j in range(col_size - 1, -1, -1):
            if final_ids[i, j] == EOS:
                repalce_idx = beam_size - (found_eos_num % beam_size) - 1
                score[repalce_idx] = final_score[i, j]
                found_eos_num += 1

                row_array[repalce_idx] = i
                col_array[repalce_idx] = j

        i -= 1

    topk_index = np.argsort(score,)[-topk:]

    trace_result = []
    trace_score = []

    for index in reversed(topk_index):
        start_i = row_array[index]
        start_j = col_array[index]
        ids = []
        for k in range(start_i, -1, -1):
            ids.append(final_ids[k, start_j])
            start_j = final_index[k, start_j]

        ids = ids[::-1]

        trace_result.append(ids)
        trace_score.append(score[index])

    return trace_result, trace_score


def load():
    """ load model for predict """
    config = model_config()
    config.vocab_size = len(open(config.vocab_path).readlines())
    final_score, final_ids, final_index = knowledge_seq2seq(config)

    final_score.persistable = True
    final_ids.persistable = True
    final_index.persistable = True

    main_program = fluid.default_main_program()

    if config.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = Executor(place)
    exe.run(framework.default_startup_program())

    fluid.io.load_params(executor=exe, dirname=config.model_path, main_program=main_program)

    processors = KnowledgeCorpus(
        data_dir=config.data_dir,
        data_prefix=config.data_prefix,
        vocab_path=config.vocab_path,
        min_len=config.min_len,
        max_len=config.max_len)

    # load dict
    id_dict_array = load_id2str_dict(config.vocab_path)

    model_handle = [exe, place, final_score, final_ids, final_index, processors, id_dict_array]
    return model_handle


def predict(model_handle, text):
    """ predict for text by model_handle """
    batch_size = 1

    [exe, place, final_score, final_ids, final_index, processors, id_dict_array] = model_handle

    data_generator = processors.preprocessing_for_lines([text], batch_size=batch_size)

    results = []
    for batch_id, data in enumerate(data_generator()):
        data_feed, sent_num = build_data_feed(data, place, batch_size=batch_size)
        out = exe.run(feed=data_feed,
                      fetch_list=[final_score.name, final_ids.name, final_index.name])

        batch_score = out[0]
        batch_ids = out[1]
        batch_pre_index = out[2]

        batch_score_arr = np.split(batch_score, batch_size, axis=1)
        batch_ids_arr = np.split(batch_ids, batch_size, axis=1)
        batch_pre_index_arr = np.split(batch_pre_index, batch_size, axis=1)

        index = 0
        for (score, ids, pre_index) in zip(batch_score_arr, batch_ids_arr, batch_pre_index_arr):
            trace_ids, trace_score = trace_fianl_result(score, ids, pre_index, topk=1, EOS=3)
            results.append(id_to_text(trace_ids[0][:-1], id_dict_array))

            index += 1
            if index >= sent_num:
                break

    return results[0]


def init_model(config, param_name_list, place):
    """ init model """
    stage = config.stage
    if stage == 0:
        for name in param_name_list:
            t = fluid.global_scope().find_var(name).get_tensor()
            init_scale = 0.05
            np_t = np.asarray(t)
            if str(name) == 'embedding':
                np_para = init_embedding(config.embed_file, config.vocab_path,
                                         init_scale, np_t.shape)
            else:
                np_para = np.random.uniform(-init_scale, init_scale, np_t.shape).astype('float32')
            t.set(np_para.astype('float32'), place)
    else:
        model_init_file = config.init_model
        try:
            model_init = np.load(model_init_file)
        except:
            print("load init model failed", model_init_file)
            raise Exception("load init model failed")

        print("load init model")
        for name in param_name_list:
            t = fluid.global_scope().find_var(name).get_tensor()
            t.set(model_init[str(name)].astype('float32'), place)

        # load opt state
        opt_state_init_file = config.init_opt_state
        if opt_state_init_file != "":
            print("begin to load opt state")
            opt_state_data = np.load(opt_state_init_file)
            for k, v in opt_state_data.items():
                t = fluid.global_scope().find_var(str(k)).get_tensor()
                t.set(v, place)
            print("set opt state finished")

    print("init model parameters finshed")


def train_loop(config,
               train_generator, valid_generator,
               main_program, inference_program,
               model_handle, param_name_list, opt_var_name_list):
    """ model train loop """
    stage = config.stage
    [exe, place, bow_loss, kl_loss, nll_loss, final_loss] = model_handle

    total_step = 0
    start_epoch = 0 if stage == 0 else config.pretrain_epoch
    end_epoch = config.pretrain_epoch if stage == 0 else config.num_epochs
    print("start end", start_epoch, end_epoch)

    best_score = float('inf')
    for epoch_idx in range(start_epoch, end_epoch):
        total_bow_loss = 0
        total_kl_loss = 0
        total_nll_loss = 0
        total_final_loss = 0
        sample_num = 0

        for batch_id, data in enumerate(train_generator()):
            data_feed = build_data_feed(data, place,
                                        batch_size=config.batch_size,
                                        is_training=True,
                                        bow_max_len=config.max_len,
                                        pretrain_epoch=epoch_idx < config.pretrain_epoch)

            if data_feed is None:
                break

            out = exe.run(main_program, feed=data_feed,
                          fetch_list=[bow_loss.name, kl_loss.name, nll_loss.name, final_loss.name])

            total_step += 1
            total_bow_loss += out[0]
            total_kl_loss += out[1]
            total_nll_loss += out[2]
            total_final_loss += out[3]
            sample_num += 1

            if batch_id > 0 and batch_id % config.log_steps == 0:
                print("epoch %d step %d | " 
                      "bow loss %0.6f kl loss %0.6f nll loss %0.6f total loss %0.6f" % \
                      (epoch_idx, batch_id,
                       total_bow_loss / sample_num, total_kl_loss / sample_num, \
                       total_nll_loss / sample_num, total_final_loss / sample_num))

                total_bow_loss = 0
                total_kl_loss = 0
                total_nll_loss = 0
                total_final_loss = 0
                sample_num = 0

            if batch_id > 0 and batch_id % config.valid_steps == 0:
                eval_bow_loss, eval_kl_loss, eval_nll_loss, eval_total_loss = \
                    vaild_loop(config, valid_generator, inference_program, model_handle)
                # save model
                if stage != 0:
                    param_path = config.save_dir + "/" + str(total_step)
                    fluid.io.save_params(executor=exe, dirname=param_path,
                                         main_program=main_program)

                    if eval_nll_loss < best_score:
                        # save to best
                        best_model_path = config.save_dir + "/best_model"
                        print("save to best", eval_nll_loss, best_model_path)
                        fluid.io.save_params(executor=exe, dirname=best_model_path,
                                             main_program=main_program)
                        best_score = eval_nll_loss

        eval_bow_loss, eval_kl_loss, eval_nll_loss, eval_total_loss = \
            vaild_loop(config, valid_generator, inference_program, model_handle)

        if stage != 0:
            param_path = config.save_dir + "/" + str(total_step)
            fluid.io.save_params(executor=exe, dirname=param_path,
                                 main_program=main_program)
            if eval_nll_loss < best_score:
                best_model_path = config.save_dir + "/best_model"
                print("save to best", eval_nll_loss, best_model_path)
                fluid.io.save_params(executor=exe, dirname=best_model_path,
                                     main_program=main_program)
                best_score = eval_nll_loss

    if stage == 0:
        # save last model and opt_stat to npz for next stage init
        save_model_file = config.save_dir + "/model_stage_0"
        save_opt_state_file = config.save_dir + "/opt_state_stage_0"

        model_stage_0 = {}
        for name in param_name_list:
            t = np.asarray(fluid.global_scope().find_var(name).get_tensor())
            model_stage_0[name] = t
        np.savez(save_model_file, **model_stage_0)

        opt_state_stage_0 = {}
        for name in opt_var_name_list:
            t_data = np.asarray(fluid.global_scope().find_var(name).get_tensor())
            opt_state_stage_0[name] = t_data
        np.savez(save_opt_state_file, **opt_state_stage_0)


def vaild_loop(config, valid_generator, inference_program, model_handle):
    """ model vaild loop """
    [exe, place, bow_loss, kl_loss, nll_loss, final_loss] = model_handle
    valid_num = 0.0
    total_valid_bow_loss = 0.0
    total_valid_kl_loss = 0.0
    total_valid_nll_loss = 0.0
    total_valid_final_loss = 0.0
    for batch_id, data in enumerate(valid_generator()):
        data_feed = build_data_feed(data, place,
                                    batch_size=config.batch_size,
                                    is_training=True,
                                    bow_max_len=config.max_len,
                                    pretrain_epoch=False)

        if data_feed is None:
            continue

        val_fetch_outs = \
            exe.run(inference_program,
                    feed=data_feed,
                    fetch_list=[bow_loss.name, kl_loss.name, nll_loss.name, final_loss.name])

        total_valid_bow_loss += val_fetch_outs[0] * config.batch_size
        total_valid_kl_loss += val_fetch_outs[1] * config.batch_size
        total_valid_nll_loss += val_fetch_outs[2] * config.batch_size
        total_valid_final_loss += val_fetch_outs[3] * config.batch_size
        valid_num += config.batch_size

    print("valid dataset: bow loss %0.6f kl loss %0.6f nll loss %0.6f total loss %0.6f" % \
          (total_valid_bow_loss / valid_num, total_valid_kl_loss / valid_num, \
           total_valid_nll_loss / valid_num, total_valid_final_loss / valid_num))

    return [total_valid_bow_loss / valid_num, total_valid_kl_loss / valid_num, \
           total_valid_nll_loss / valid_num, total_valid_final_loss / valid_num]


def test(config):
    """ test """
    batch_size = config.batch_size
    config.vocab_size = len(open(config.vocab_path).readlines())
    final_score, final_ids, final_index = knowledge_seq2seq(config)

    final_score.persistable = True
    final_ids.persistable = True
    final_index.persistable = True

    main_program = fluid.default_main_program()

    if config.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = Executor(place)
    exe.run(framework.default_startup_program())

    fluid.io.load_params(executor=exe, dirname=config.model_path,
                         main_program=main_program)
    print("laod params finsihed")

    # test data generator
    processors = KnowledgeCorpus(
        data_dir=config.data_dir,
        data_prefix=config.data_prefix,
        vocab_path=config.vocab_path,
        min_len=config.min_len,
        max_len=config.max_len)
    test_generator = processors.data_generator(
        batch_size=config.batch_size,
        phase="test",
        shuffle=False)

    # load dict
    id_dict_array = load_id2str_dict(config.vocab_path)

    out_file = config.output
    fout = open(out_file, 'w')
    for batch_id, data in enumerate(test_generator()):
        data_feed, sent_num = build_data_feed(data, place, batch_size=batch_size)

        if data_feed is None:
            break

        out = exe.run(feed=data_feed,
                      fetch_list=[final_score.name, final_ids.name, final_index.name])

        batch_score = out[0]
        batch_ids = out[1]
        batch_pre_index = out[2]

        batch_score_arr = np.split(batch_score, batch_size, axis=1)
        batch_ids_arr = np.split(batch_ids, batch_size, axis=1)
        batch_pre_index_arr = np.split(batch_pre_index, batch_size, axis=1)

        index = 0
        for (score, ids, pre_index) in zip(batch_score_arr, batch_ids_arr, batch_pre_index_arr):
            trace_ids, trace_score = trace_fianl_result(score, ids, pre_index, topk=1, EOS=3)
            fout.write(id_to_text(trace_ids[0][:-1], id_dict_array))
            fout.write('\n')

            index += 1
            if index >= sent_num:
                break

    fout.close()


def train(config):
    """ model training """
    config.vocab_size = len(open(config.vocab_path).readlines())
    bow_loss, kl_loss, nll_loss, final_loss= knowledge_seq2seq(config)

    bow_loss.persistable = True
    kl_loss.persistable = True
    nll_loss.persistable = True
    final_loss.persistable = True
    
    main_program = fluid.default_main_program()
    inference_program = fluid.default_main_program().clone(for_test=True)

    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=config.grad_clip))
    optimizer = fluid.optimizer.Adam(learning_rate=config.lr)

    if config.stage == 0:
        print("stage 0")
        optimizer.minimize(bow_loss)
    else:
        print("stage 1")
        optimizer.minimize(final_loss)

    opt_var_name_list = optimizer.get_opti_var_name_list()

    if config.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = Executor(place)
    exe.run(framework.default_startup_program())

    param_list = main_program.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]

    init_model(config, param_name_list, place)

    processors = KnowledgeCorpus(
                        data_dir=config.data_dir,
                        data_prefix=config.data_prefix,
                        vocab_path=config.vocab_path,
                        min_len=config.min_len,
                        max_len=config.max_len)
    train_generator = processors.data_generator(
                        batch_size=config.batch_size,
                        phase="train",
                        shuffle=True)
    valid_generator = processors.data_generator(
                        batch_size=config.batch_size,
                        phase="dev",
                        shuffle=False)

    model_handle = [exe, place, bow_loss, kl_loss, nll_loss, final_loss]

    train_loop(config,
               train_generator, valid_generator,
               main_program, inference_program,
               model_handle, param_name_list, opt_var_name_list)


if __name__ == "__main__":
    config = model_config()
    run_type = config.run_type
    
    if run_type == "train":
        train(config)
    elif run_type == "test":
        test(config)
