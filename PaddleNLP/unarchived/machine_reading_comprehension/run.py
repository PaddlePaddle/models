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
import random
import json
import six
import multiprocessing

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
sys.path.append('..')

from args import *
import rc_model
from dataset import BRCDataset
import logging
import pickle
from utils import normalize
from utils import compute_bleu_rouge
from vocab import Vocab


def prepare_batch_input(insts, args):
    batch_size = len(insts['raw_data'])
    inst_num = len(insts['passage_num'])
    if batch_size != inst_num:
        print("data error %d, %d" % (batch_size, inst_num))
        return None
    new_insts = []

    passage_idx = 0
    for i in range(batch_size):
        p_len = 0
        p_id = []
        p_ids = []
        q_ids = []
        q_id = []
        p_id_r = []
        p_ids_r = []
        q_ids_r = []
        q_id_r = []

        for j in range(insts['passage_num'][i]):
            p_ids.append(insts['passage_token_ids'][passage_idx + j])
            p_id = p_id + insts['passage_token_ids'][passage_idx + j]
            q_ids.append(insts['question_token_ids'][passage_idx + j])
            q_id = q_id + insts['question_token_ids'][passage_idx + j]

        passage_idx += insts['passage_num'][i]
        p_len = len(p_id)

        def _get_label(idx, ref_len):
            ret = [0.0] * ref_len
            if idx >= 0 and idx < ref_len:
                ret[idx] = 1.0
            return [[x] for x in ret]

        start_label = _get_label(insts['start_id'][i], p_len)
        end_label = _get_label(insts['end_id'][i], p_len)
        new_inst = [q_ids, start_label, end_label, p_ids, q_id]
        new_insts.append(new_inst)
    return new_insts


def batch_reader(batch_list, args):
    res = []
    for batch in batch_list:
        res.append(prepare_batch_input(batch, args))
    return res


def read_multiple(reader, count, clip_last=True):
    """
    Stack data from reader for multi-devices.
    """

    def __impl__():
        res = []
        for item in reader():
            res.append(item)
            if len(res) == count:
                yield res
                res = []
        if len(res) == count:
            yield res
        elif not clip_last:
            data = []
            for item in res:
                data += item
            if len(data) > count:
                inst_num_per_part = len(data) // count
                yield [
                    data[inst_num_per_part * i:inst_num_per_part * (i + 1)]
                    for i in range(count)
                ]

    return __impl__


def LodTensor_Array(lod_tensor):
    lod = lod_tensor.lod()
    array = np.array(lod_tensor)
    new_array = []
    for i in range(len(lod[0]) - 1):
        new_array.append(array[lod[0][i]:lod[0][i + 1]])
    return new_array


def print_para(train_prog, train_exe, logger, args):
    """Print para info for debug purpose"""
    if args.para_print:
        param_list = train_prog.block(0).all_parameters()
        param_name_list = [p.name for p in param_list]
        num_sum = 0
        for p_name in param_name_list:
            p_array = np.array(train_exe.scope.find_var(p_name).get_tensor())
            param_num = np.prod(p_array.shape)
            num_sum = num_sum + param_num
            logger.info(
                "param: {0},  mean={1}  max={2}  min={3}  num={4} {5}".format(
                    p_name,
                    p_array.mean(),
                    p_array.max(), p_array.min(), p_array.shape, param_num))
        logger.info("total param num: {0}".format(num_sum))


def find_best_answer_for_passage(start_probs, end_probs, passage_len):
    """
    Finds the best answer with the maximum start_prob * end_prob from a single passage
    """
    if passage_len is None:
        passage_len = len(start_probs)
    else:
        passage_len = min(len(start_probs), passage_len)
    best_start, best_end, max_prob = -1, -1, 0
    for start_idx in range(passage_len):
        for ans_len in range(args.max_a_len):
            end_idx = start_idx + ans_len
            if end_idx >= passage_len:
                continue
            prob = start_probs[start_idx] * end_probs[end_idx]
            if prob > max_prob:
                best_start = start_idx
                best_end = end_idx
                max_prob = prob
    return (best_start, best_end), max_prob


def find_best_answer_for_inst(sample,
                              start_prob,
                              end_prob,
                              inst_lod,
                              para_prior_scores=(0.44, 0.23, 0.15, 0.09, 0.07)):
    """
    Finds the best answer for a sample given start_prob and end_prob for each position.
    This will call find_best_answer_for_passage because there are multiple passages in a sample
    """
    best_p_idx, best_span, best_score = None, None, 0
    for p_idx, passage in enumerate(sample['passages']):
        if p_idx >= args.max_p_num:
            continue
        if len(start_prob) != len(end_prob):
            logger.info('error: {}'.format(sample['question']))
            continue
        passage_start = inst_lod[p_idx] - inst_lod[0]
        passage_end = inst_lod[p_idx + 1] - inst_lod[0]
        passage_len = passage_end - passage_start
        passage_len = min(args.max_p_len, len(passage['passage_tokens']))
        answer_span, score = find_best_answer_for_passage(
            start_prob[passage_start:passage_end],
            end_prob[passage_start:passage_end], passage_len)
        if para_prior_scores is not None:
            # the Nth prior score = the Number of training samples whose gold answer comes
            #  from the Nth paragraph / the number of the training samples
            score *= para_prior_scores[p_idx]
        if score > best_score:
            best_score = score
            best_p_idx = p_idx
            best_span = answer_span
    if best_p_idx is None or best_span is None:
        best_answer = ''
    else:
        best_answer = ''.join(sample['passages'][best_p_idx]['passage_tokens'][
            best_span[0]:best_span[1] + 1])
    return best_answer, best_span


def validation(inference_program, avg_cost, s_probs, e_probs, match, feed_order,
               place, dev_count, vocab, brc_data, logger, args):
    """
    do inference with given inference_program
    """
    parallel_executor = fluid.ParallelExecutor(
        main_program=inference_program,
        use_cuda=bool(args.use_gpu),
        loss_name=avg_cost.name)
    print_para(inference_program, parallel_executor, logger, args)

    # Use test set as validation each pass
    total_loss = 0.0
    count = 0
    n_batch_cnt = 0
    n_batch_loss = 0.0
    pred_answers, ref_answers = [], []
    val_feed_list = [
        inference_program.global_block().var(var_name)
        for var_name in feed_order
    ]
    val_feeder = fluid.DataFeeder(val_feed_list, place)
    pad_id = vocab.get_id(vocab.pad_token)
    dev_reader = lambda:brc_data.gen_mini_batches('dev', args.batch_size, pad_id, shuffle=False)
    dev_reader = read_multiple(dev_reader, dev_count)

    for batch_id, batch_list in enumerate(dev_reader(), 1):
        feed_data = batch_reader(batch_list, args)
        val_fetch_outs = parallel_executor.run(
            feed=list(val_feeder.feed_parallel(feed_data, dev_count)),
            fetch_list=[avg_cost.name, s_probs.name, e_probs.name, match.name],
            return_numpy=False)
        total_loss += np.array(val_fetch_outs[0]).sum()
        start_probs_m = LodTensor_Array(val_fetch_outs[1])
        end_probs_m = LodTensor_Array(val_fetch_outs[2])
        match_lod = val_fetch_outs[3].lod()
        count += len(np.array(val_fetch_outs[0]))

        n_batch_cnt += len(np.array(val_fetch_outs[0]))
        n_batch_loss += np.array(val_fetch_outs[0]).sum()
        log_every_n_batch = args.log_interval
        if log_every_n_batch > 0 and batch_id % log_every_n_batch == 0:
            logger.info('Average dev loss from batch {} to {} is {}'.format(
                batch_id - log_every_n_batch + 1, batch_id, "%.10f" % (
                    n_batch_loss / n_batch_cnt)))
            n_batch_loss = 0.0
            n_batch_cnt = 0
        batch_offset = 0
        for idx, batch in enumerate(batch_list):
            #one batch
            batch_size = len(batch['raw_data'])
            batch_range = match_lod[0][batch_offset:batch_offset + batch_size +
                                       1]
            batch_lod = [[batch_range[x], batch_range[x + 1]]
                         for x in range(len(batch_range[:-1]))]
            start_prob_batch = start_probs_m[batch_offset:batch_offset +
                                             batch_size + 1]
            end_prob_batch = end_probs_m[batch_offset:batch_offset + batch_size
                                         + 1]
            for sample, start_prob_inst, end_prob_inst, inst_range in zip(
                    batch['raw_data'], start_prob_batch, end_prob_batch,
                    batch_lod):
                #one instance
                inst_lod = match_lod[1][inst_range[0]:inst_range[1] + 1]
                best_answer, best_span = find_best_answer_for_inst(
                    sample, start_prob_inst, end_prob_inst, inst_lod)
                pred = {
                    'question_id': sample['question_id'],
                    'question_type': sample['question_type'],
                    'answers': [best_answer],
                    'entity_answers': [[]],
                    'yesno_answers': []
                }
                pred_answers.append(pred)
                if 'answers' in sample:
                    ref = {
                        'question_id': sample['question_id'],
                        'question_type': sample['question_type'],
                        'answers': sample['answers'],
                        'entity_answers': [[]],
                        'yesno_answers': []
                    }
                    ref_answers.append(ref)
            batch_offset = batch_offset + batch_size

    result_dir = args.result_dir
    result_prefix = args.result_name
    if result_dir is not None and result_prefix is not None:
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        result_file = os.path.join(result_dir, result_prefix + '.json')
        with open(result_file, 'w') as fout:
            for pred_answer in pred_answers:
                fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')
        logger.info('Saving {} results to {}'.format(result_prefix,
                                                     result_file))

    ave_loss = 1.0 * total_loss / count
    # compute the bleu and rouge scores if reference answers is provided
    if len(ref_answers) > 0:
        pred_dict, ref_dict = {}, {}
        for pred, ref in zip(pred_answers, ref_answers):
            question_id = ref['question_id']
            if len(ref['answers']) > 0:
                pred_dict[question_id] = normalize(pred['answers'])
                ref_dict[question_id] = normalize(ref['answers'])
        bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
    else:
        bleu_rouge = None
    return ave_loss, bleu_rouge


def l2_loss(train_prog):
    param_list = train_prog.block(0).all_parameters()
    para_sum = []
    for para in param_list:
        para_mul = fluid.layers.elementwise_mul(x=para, y=para, axis=0)
        para_sum.append(fluid.layers.reduce_sum(input=para_mul, dim=None))
    return fluid.layers.sums(para_sum) * 0.5


def train(logger, args):
    """train a model"""
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        if six.PY2:
            vocab = pickle.load(fin)
        else:
            vocab = pickle.load(fin, encoding='bytes')
        logger.info('vocab size is {} and embed dim is {}'.format(vocab.size(
        ), vocab.embed_dim))
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.trainset, args.devset)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Initialize the model...')

    if not args.use_gpu:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    else:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()

    # build model
    main_program = fluid.Program()
    startup_prog = fluid.Program()
    if args.enable_ce:
        main_program.random_seed = args.random_seed
        startup_prog.random_seed = args.random_seed
    with fluid.program_guard(main_program, startup_prog):
        with fluid.unique_name.guard():
            avg_cost, s_probs, e_probs, match, feed_order = rc_model.rc_model(
                args.hidden_size, vocab, args)
            # clone from default main program and use it as the validation program
            inference_program = main_program.clone(for_test=True)

            # build optimizer
            if args.optim == 'sgd':
                optimizer = fluid.optimizer.SGD(
                    learning_rate=args.learning_rate)
            elif args.optim == 'adam':
                optimizer = fluid.optimizer.Adam(
                    learning_rate=args.learning_rate)
            elif args.optim == 'rprop':
                optimizer = fluid.optimizer.RMSPropOptimizer(
                    learning_rate=args.learning_rate)
            else:
                logger.error('Unsupported optimizer: {}'.format(args.optim))
                exit(-1)
            if args.weight_decay > 0.0:
                obj_func = avg_cost + args.weight_decay * l2_loss(main_program)
                optimizer.minimize(obj_func)
            else:
                obj_func = avg_cost
                optimizer.minimize(obj_func)

            # initialize parameters
            place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
            exe = Executor(place)
            if args.load_dir:
                logger.info('load from {}'.format(args.load_dir))
                fluid.io.load_persistables(
                    exe, args.load_dir, main_program=main_program)
            else:
                exe.run(startup_prog)
                embedding_para = fluid.global_scope().find_var(
                    'embedding_para').get_tensor()
                embedding_para.set(vocab.embeddings.astype(np.float32), place)

            # prepare data
            feed_list = [
                main_program.global_block().var(var_name)
                for var_name in feed_order
            ]
            feeder = fluid.DataFeeder(feed_list, place)

            logger.info('Training the model...')
            parallel_executor = fluid.ParallelExecutor(
                main_program=main_program,
                use_cuda=bool(args.use_gpu),
                loss_name=avg_cost.name)
            print_para(main_program, parallel_executor, logger, args)

            for pass_id in range(1, args.pass_num + 1):
                pass_start_time = time.time()
                pad_id = vocab.get_id(vocab.pad_token)
                if args.enable_ce:
                    train_reader = lambda:brc_data.gen_mini_batches('train', args.batch_size, pad_id, shuffle=False)
                else:
                    train_reader = lambda:brc_data.gen_mini_batches('train', args.batch_size, pad_id, shuffle=True)
                train_reader = read_multiple(train_reader, dev_count)
                log_every_n_batch, n_batch_loss = args.log_interval, 0
                total_num, total_loss = 0, 0
                for batch_id, batch_list in enumerate(train_reader(), 1):
                    feed_data = batch_reader(batch_list, args)
                    fetch_outs = parallel_executor.run(
                        feed=list(feeder.feed_parallel(feed_data, dev_count)),
                        fetch_list=[obj_func.name],
                        return_numpy=False)
                    cost_train = np.array(fetch_outs[0]).mean()
                    total_num += args.batch_size * dev_count
                    n_batch_loss += cost_train
                    total_loss += cost_train * args.batch_size * dev_count

                    if args.enable_ce and batch_id >= 100:
                        break
                    if log_every_n_batch > 0 and batch_id % log_every_n_batch == 0:
                        print_para(main_program, parallel_executor, logger,
                                   args)
                        logger.info(
                            'Average loss from batch {} to {} is {}'.format(
                                batch_id - log_every_n_batch + 1, batch_id,
                                "%.10f" % (n_batch_loss / log_every_n_batch)))
                        n_batch_loss = 0
                    if args.dev_interval > 0 and batch_id % args.dev_interval == 0:
                        if brc_data.dev_set is not None:
                            eval_loss, bleu_rouge = validation(
                                inference_program, avg_cost, s_probs, e_probs,
                                match, feed_order, place, dev_count, vocab,
                                brc_data, logger, args)
                            logger.info('Dev eval loss {}'.format(eval_loss))
                            logger.info('Dev eval result: {}'.format(
                                bleu_rouge))
                pass_end_time = time.time()
                time_consumed = pass_end_time - pass_start_time
                logger.info('epoch: {0}, epoch_time_cost: {1:.2f}'.format(
                    pass_id, time_consumed))
                logger.info('Evaluating the model after epoch {}'.format(
                    pass_id))
                if brc_data.dev_set is not None:
                    eval_loss, bleu_rouge = validation(
                        inference_program, avg_cost, s_probs, e_probs, match,
                        feed_order, place, dev_count, vocab, brc_data, logger,
                        args)
                    logger.info('Dev eval loss {}'.format(eval_loss))
                    logger.info('Dev eval result: {}'.format(bleu_rouge))
                else:
                    logger.warning(
                        'No dev set is loaded for evaluation in the dataset!')

                logger.info('Average train loss for epoch {} is {}'.format(
                    pass_id, "%.10f" % (1.0 * total_loss / total_num)))

                if pass_id % args.save_interval == 0:
                    model_path = os.path.join(args.save_dir, str(pass_id))
                    if not os.path.isdir(model_path):
                        os.makedirs(model_path)

                    fluid.io.save_persistables(
                        executor=exe,
                        dirname=model_path,
                        main_program=main_program)
                if args.enable_ce:  # For CE
                    print("kpis\ttrain_cost_card%d\t%f" %
                          (dev_count, total_loss / total_num))
                    if brc_data.dev_set is not None:
                        print("kpis\ttest_cost_card%d\t%f" %
                              (dev_count, eval_loss))
                    print("kpis\ttrain_duration_card%d\t%f" %
                          (dev_count, time_consumed))


def evaluate(logger, args):
    """evaluate a specific model using devset"""
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
        logger.info('vocab size is {} and embed dim is {}'.format(vocab.size(
        ), vocab.embed_dim))
    brc_data = BRCDataset(
        args.max_p_num, args.max_p_len, args.max_q_len, dev_files=args.devset)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Initialize the model...')

    # build model
    main_program = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(main_program, startup_prog):
        with fluid.unique_name.guard():
            avg_cost, s_probs, e_probs, match, feed_order = rc_model.rc_model(
                args.hidden_size, vocab, args)
            # initialize parameters
            if not args.use_gpu:
                place = fluid.CPUPlace()
                dev_count = int(
                    os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
            else:
                place = fluid.CUDAPlace(0)
                dev_count = fluid.core.get_cuda_device_count()

            exe = Executor(place)
            if args.load_dir:
                logger.info('load from {}'.format(args.load_dir))
                fluid.io.load_persistables(
                    exe, args.load_dir, main_program=main_program)
            else:
                logger.error('No model file to load ...')
                return

            inference_program = main_program.clone(for_test=True)
            eval_loss, bleu_rouge = validation(
                inference_program, avg_cost, s_probs, e_probs, match,
                feed_order, place, dev_count, vocab, brc_data, logger, args)
            logger.info('Dev eval loss {}'.format(eval_loss))
            logger.info('Dev eval result: {}'.format(bleu_rouge))
            logger.info('Predicted answers are saved to {}'.format(
                os.path.join(args.result_dir)))


def predict(logger, args):
    """do inference on the test dataset """
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
        logger.info('vocab size is {} and embed dim is {}'.format(vocab.size(
        ), vocab.embed_dim))
    brc_data = BRCDataset(
        args.max_p_num, args.max_p_len, args.max_q_len, dev_files=args.testset)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Initialize the model...')

    # build model
    main_program = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(main_program, startup_prog):
        with fluid.unique_name.guard():
            avg_cost, s_probs, e_probs, match, feed_order = rc_model.rc_model(
                args.hidden_size, vocab, args)
            # initialize parameters
            if not args.use_gpu:
                place = fluid.CPUPlace()
                dev_count = int(
                    os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
            else:
                place = fluid.CUDAPlace(0)
                dev_count = fluid.core.get_cuda_device_count()

            exe = Executor(place)
            if args.load_dir:
                logger.info('load from {}'.format(args.load_dir))
                fluid.io.load_persistables(
                    exe, args.load_dir, main_program=main_program)
            else:
                logger.error('No model file to load ...')
                return

            inference_program = main_program.clone(for_test=True)
            eval_loss, bleu_rouge = validation(
                inference_program, avg_cost, s_probs, e_probs, match,
                feed_order, place, dev_count, vocab, brc_data, logger, args)


def prepare(logger, args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger.info('Checking the data files...')
    for data_path in args.trainset + args.devset + args.testset:
        assert os.path.exists(data_path), '{} file does not exist.'.format(
            data_path)
    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.save_dir, args.result_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.trainset, args.devset, args.testset)
    vocab = Vocab(lower=True)
    for word in brc_data.word_iter('train'):
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(
        filtered_num, vocab.size()))

    logger.info('Assigning embeddings...')
    vocab.randomly_init_embeddings(args.embed_size)

    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('Done with preparing!')


if __name__ == '__main__':
    args = parse_args()

    if args.enable_ce:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    args = parse_args()
    logger.info('Running with args : {}'.format(args))
    if args.prepare:
        prepare(logger, args)
    if args.train:
        train(logger, args)
    if args.evaluate:
        evaluate(logger, args)
    if args.predict:
        predict(logger, args)
