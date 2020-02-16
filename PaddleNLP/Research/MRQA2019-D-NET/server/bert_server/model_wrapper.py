#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BERT (PaddlePaddle) model wrapper"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import collections
import multiprocessing
import argparse
import numpy as np
import paddle.fluid as fluid
from task_reader.mrqa import DataProcessor, get_answers

ema_decay = 0.9999
verbose = False
max_seq_len = 512
max_query_length = 64
max_answer_length = 30
in_tokens = False
do_lower_case = False
doc_stride = 128
n_best_size = 20
use_cuda = True

class BertModelWrapper():
    """
    Wrap a tnet model
     the basic processes include input checking, preprocessing, calling tf-serving
     and postprocessing
    """
    def __init__(self, model_dir):
        """ """
        if use_cuda:
            place = fluid.CUDAPlace(0)
            dev_count = fluid.core.get_cuda_device_count()
        else:
            place = fluid.CPUPlace()
            dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
        self.exe = fluid.Executor(place)

        self.bert_preprocessor = DataProcessor(
            vocab_path=os.path.join(model_dir, 'vocab.txt'),
            do_lower_case=do_lower_case,
            max_seq_length=max_seq_len,
            in_tokens=in_tokens,
            doc_stride=doc_stride,
            max_query_length=max_query_length)

        self.inference_program, self.feed_target_names, self.fetch_targets = \
            fluid.io.load_inference_model(dirname=model_dir, executor=self.exe)

    def preprocessor(self, samples, batch_size, examples_start_id, features_start_id):
        """Preprocess the input samples, including word seg, padding, token to ids"""
        # Tokenization and paragraph padding
        examples, features, batch = self.bert_preprocessor.data_generator(
            samples, batch_size, max_len=max_seq_len, examples_start_id=examples_start_id, features_start_id=features_start_id)
        self.samples = samples
        return examples, features, batch

    def call_mrc(self, batch, squeeze_dim0=False, return_list=False):
        """MRC"""
        if squeeze_dim0 and return_list:
            raise ValueError("squeeze_dim0 only work for dict-type return value.")
        src_ids = batch[0]
        pos_ids = batch[1]
        sent_ids = batch[2]
        input_mask = batch[3]
        unique_id = batch[4]
        feed_dict = {
            self.feed_target_names[0]: src_ids,
            self.feed_target_names[1]: pos_ids,
            self.feed_target_names[2]: sent_ids,
            self.feed_target_names[3]: input_mask,
            self.feed_target_names[4]: unique_id
        }
        
        np_unique_ids, np_start_logits, np_end_logits, np_num_seqs = \
            self.exe.run(self.inference_program, feed=feed_dict, fetch_list=self.fetch_targets)

        if len(np_unique_ids) == 1 and squeeze_dim0:
            np_unique_ids = np_unique_ids[0]
            np_start_logits = np_start_logits[0]
            np_end_logits = np_end_logits[0]

        if return_list:
            mrc_results = [{'unique_ids': id, 'start_logits': st, 'end_logits': end} 
                            for id, st, end in zip(np_unique_ids, np_start_logits, np_end_logits)]
        else:
            mrc_results = {
                'unique_ids': np_unique_ids,
                'start_logits': np_start_logits,
                'end_logits': np_end_logits,
            }
        return mrc_results

    def postprocessor(self, examples, features, mrc_results):
        """Extract answer
         batch: [examples, features] from preprocessor
         mrc_results: model results from call_mrc. if mrc_results is list, each element of which is a size=1 batch.
        """
        RawResult = collections.namedtuple("RawResult",
                                           ["unique_id", "start_logits", "end_logits"])
        results = []
        if isinstance(mrc_results, list):
            for res in mrc_results:
                unique_id = res['unique_ids'][0]
                start_logits = [float(x) for x in res['start_logits'].flat]
                end_logits = [float(x) for x in res['end_logits'].flat]
                results.append(
                    RawResult(
                        unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits))
        else:
            assert isinstance(mrc_results, dict)
            for idx in range(mrc_results['unique_ids'].shape[0]):
                unique_id = int(mrc_results['unique_ids'][idx])
                start_logits = [float(x) for x in mrc_results['start_logits'][idx].flat]
                end_logits = [float(x) for x in mrc_results['end_logits'][idx].flat]
                results.append(
                    RawResult(
                        unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits))
        
        answers = get_answers(
            examples, features, results, n_best_size,
            max_answer_length, do_lower_case, verbose)
        return answers

