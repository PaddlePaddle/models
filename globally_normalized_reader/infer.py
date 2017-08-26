#!/usr/bin/env python
#coding=utf-8
import os
import sys
import gzip
import logging
import numpy as np
import pdb

import paddle.v2 as paddle
from paddle.v2.layer import parse_network
import reader

from model import GNR
from train import choose_samples
from config import ModelConfig, TrainerConfig

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def load_reverse_dict(dict_file):
    word_dict = {}
    with open(dict_file, "r") as fin:
        for idx, line in enumerate(fin):
            word_dict[idx] = line.strip()
    return word_dict


def parse_one_sample(raw_input_doc, sub_sen_scores, selected_sentence,
                     start_span_scores, selected_starts, end_span_scores,
                     selected_ends):
    assert len(raw_input_doc) == sub_sen_scores.shape[0]
    beam_size = selected_sentence.shape[1]

    all_searched_ans = []
    for i in xrange(selected_ends.shape[0]):
        for j in xrange(selected_ends.shape[1]):
            if selected_ends[i][j] == -1.: break
            all_searched_ans.append({
                'score': end_span_scores[int(selected_ends[i][j])],
                'sentence_pos': -1,
                'start_span_pos': -1,
                'end_span_pos': int(selected_ends[i][j]),
                'parent_ids_in_prev_beam': i
            })

    for path in all_searched_ans:
        row_id = path['parent_ids_in_prev_beam'] / beam_size
        col_id = path['parent_ids_in_prev_beam'] % beam_size
        path['start_span_pos'] = int(selected_starts[row_id][col_id])
        path['score'] += start_span_scores[path['start_span_pos']]
        path['parent_ids_in_prev_beam'] = row_id

    for path in all_searched_ans:
        row_id = path['parent_ids_in_prev_beam'] / beam_size
        col_id = path['parent_ids_in_prev_beam'] % beam_size
        path['sentence_pos'] = int(selected_sentence[row_id][col_id])
        path['score'] += sub_sen_scores[path['sentence_pos']]

    all_searched_ans.sort(key=lambda x: x['score'], reverse=True)
    return all_searched_ans


def infer_a_batch(inferer, test_batch, ids_2_word, out_layer_count):
    outs = inferer.infer(input=test_batch, flatten_result=False, field="value")

    for test_sample in test_batch:
        query_word = [ids_2_word[ids] for ids in test_sample[0]]
        print("query\n\t%s\ndocument" % (" ".join(query_word)))

        # iterate over each word of in document
        for i, sentence in enumerate(test_sample[1]):
            sen_word = [ids_2_word[ids] for ids in sentence]
            print("%d\t%s" % (i, " ".join(sen_word)))
        print("gold\t[%d %d %d]" %
              (test_sample[3], test_sample[4], test_sample[5]))

        ans = parse_one_sample(test_sample[1], *outs)[0]
        ans_ids = test_sample[1][ans['sentence_pos']][ans['start_span_pos']:ans[
            'start_span_pos'] + ans['end_span_pos']]
        ans_str = " ".join([ids_2_word[ids] for ids in ans_ids])
        print("searched answer\t[%d %d %d]\n\t%s" %
              (ans['sentence_pos'], ans['start_span_pos'], ans['end_span_pos'],
               ans_str))


def infer(model_path, data_dir, test_batch_size, config):
    assert os.path.exists(model_path), "The model does not exist."
    paddle.init(use_gpu=False, trainer_count=1)

    ids_2_word = load_reverse_dict(config.dict_path)

    outputs = GNR(config, is_infer=True)

    # load the trained models
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(model_path, "r"))
    inferer = paddle.inference.Inference(
        output_layer=outputs, parameters=parameters)

    _, valid_samples = choose_samples(data_dir)
    test_reader = reader.data_reader(valid_samples, is_train=False)

    test_batch = []
    for i, item in enumerate(test_reader()):
        test_batch.append(item)
        if len(test_batch) == test_batch_size:
            infer_a_batch(inferer, test_batch, ids_2_word, len(outputs))
            test_batch = []

    if len(test_batch):
        infer_a_batch(inferer, test_batch, ids_2_word, len(outputs))
        test_batch = []


if __name__ == "__main__":
    infer("models/pass_00003.tar.gz", TrainerConfig.data_dir,
          TrainerConfig.test_batch_size, ModelConfig)
