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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import time
import argparse
import numpy as np
import json
import multiprocessing

import paddle
import logging
import paddle.fluid as fluid

from model.ernie import ErnieModel

log = logging.getLogger(__name__)


def create_model(args, pyreader_name, ernie_config):
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, args.num_labels], [-1, 1], [-1, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1]],
        dtypes=[
            'int64', 'int64', 'int64', 'int64', 'float32', 'float32', 'int64',
            'int64', 'int64', 'int64'
        ],
        lod_levels=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, sent_ids, pos_ids, task_ids, input_mask, labels, seq_lens,
     example_index, tok_to_orig_start_index,
     tok_to_orig_end_index) = fluid.layers.read_file(pyreader)

    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        task_ids=task_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16)

    enc_out = ernie.get_sequence_output()
    enc_out = fluid.layers.dropout(
        x=enc_out, dropout_prob=0.1, dropout_implementation="upscale_in_train")
    logits = fluid.layers.fc(
        input=enc_out,
        size=args.num_labels,
        num_flatten_dims=2,
        param_attr=fluid.ParamAttr(
            name="cls_seq_label_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_seq_label_out_b",
            initializer=fluid.initializer.Constant(0.)))
    logits = fluid.layers.sigmoid(logits)

    lod_labels = fluid.layers.sequence_unpad(labels, seq_lens)
    lod_logits = fluid.layers.sequence_unpad(logits, seq_lens)
    lod_tok_to_orig_start_index = fluid.layers.sequence_unpad(
        tok_to_orig_start_index, seq_lens)
    lod_tok_to_orig_end_index = fluid.layers.sequence_unpad(
        tok_to_orig_end_index, seq_lens)

    labels = fluid.layers.flatten(labels, axis=2)
    logits = fluid.layers.flatten(logits, axis=2)
    input_mask = fluid.layers.flatten(input_mask, axis=2)

    # calculate loss
    log_logits = fluid.layers.log(logits)
    log_logits_neg = fluid.layers.log(1 - logits)
    ce_loss = 0. - labels * log_logits - (1 - labels) * log_logits_neg

    ce_loss = fluid.layers.reduce_mean(ce_loss, dim=1, keep_dim=True)
    ce_loss = ce_loss * input_mask
    loss = fluid.layers.mean(x=ce_loss)

    graph_vars = {
        "inputs": src_ids,
        "loss": loss,
        "seqlen": seq_lens,
        "lod_logit": lod_logits,
        "lod_label": lod_labels,
        "example_index": example_index,
        "tok_to_orig_start_index": lod_tok_to_orig_start_index,
        "tok_to_orig_end_index": lod_tok_to_orig_end_index
    }

    for k, v in graph_vars.items():
        v.persistable = True

    return pyreader, graph_vars


def calculate_acc(logits, labels):
    # the golden metric should be "f1" in spo level
    # but here only "accuracy" is computed during training for simplicity (provide crude view of your training status)
    # accuracy is dependent on the tagging strategy
    # for each token, the prediction is counted as correct if all its 100 labels were correctly predicted
    # for each example, the prediction is counted as correct if all its token were correctly predicted

    logits_lod = logits.lod()
    labels_lod = labels.lod()
    logits_tensor = np.array(logits)
    labels_tensor = np.array(labels)
    assert logits_lod == labels_lod

    num_total = 0
    num_correct = 0
    token_total = 0
    token_correct = 0
    for i in range(len(logits_lod[0]) - 1):
        inference_tmp = logits_tensor[logits_lod[0][i]:logits_lod[0][i + 1]]
        inference_tmp[inference_tmp >= 0.5] = 1
        inference_tmp[inference_tmp < 0.5] = 0
        label_tmp = labels_tensor[labels_lod[0][i]:labels_lod[0][i + 1]]
        num_total += 1
        if (inference_tmp == label_tmp).all():
            num_correct += 1
        for j in range(len(inference_tmp)):
            token_total += 1
            if (inference_tmp[j] == label_tmp[j]).all():
                token_correct += 1
    return num_correct, num_total, token_correct, token_total


def calculate_metric(spo_list_gt, spo_list_predict):
    # calculate golden metric precision, recall and f1
    # may be slightly different with final official evaluation on test set,
    # because more comprehensive detail is considered (e.g. alias)
    tp, fp, fn = 0, 0, 0

    for spo in spo_list_predict:
        flag = 0
        for spo_gt in spo_list_gt:
            if spo['predicate'] == spo_gt['predicate'] and spo[
                    'object'] == spo_gt['object'] and spo['subject'] == spo_gt[
                        'subject']:
                flag = 1
                tp += 1
                break
        if flag == 0:
            fp += 1
    '''
    for spo in spo_list_predict:
        if spo in spo_list_gt:
            tp += 1
        else:
            fp += 1
            '''

    fn = len(spo_list_gt) - tp
    return tp, fp, fn


def evaluate(args, examples, exe, program, pyreader, graph_vars):
    spo_label_map = json.load(open(args.spo_label_map_config))

    fetch_list = [
        graph_vars["lod_logit"].name, graph_vars["lod_label"].name,
        graph_vars["example_index"].name,
        graph_vars["tok_to_orig_start_index"].name,
        graph_vars["tok_to_orig_end_index"].name
    ]

    tp, fp, fn = 0, 0, 0

    time_begin = time.time()
    pyreader.start()
    while True:
        try:
            # prepare fetched batch data: unlod etc.
            logits, labels, example_index_list, tok_to_orig_start_index_list, tok_to_orig_end_index_list = \
                exe.run(program=program, fetch_list=fetch_list, return_numpy=False)
            example_index_list = np.array(example_index_list).astype(
                int) - 100000
            logits_lod = logits.lod()
            tok_to_orig_start_index_list_lod = tok_to_orig_start_index_list.lod(
            )
            tok_to_orig_end_index_list_lod = tok_to_orig_end_index_list.lod()
            logits_tensor = np.array(logits)
            tok_to_orig_start_index_list = np.array(
                tok_to_orig_start_index_list).flatten()
            tok_to_orig_end_index_list = np.array(
                tok_to_orig_end_index_list).flatten()

            # perform evaluation
            for i in range(len(logits_lod[0]) - 1):
                # prepare prediction results for each example
                example_index = example_index_list[i]
                example = examples[example_index]
                tok_to_orig_start_index = tok_to_orig_start_index_list[
                    tok_to_orig_start_index_list_lod[0][
                        i]:tok_to_orig_start_index_list_lod[0][i + 1] - 2]
                tok_to_orig_end_index = tok_to_orig_end_index_list[
                    tok_to_orig_end_index_list_lod[0][
                        i]:tok_to_orig_end_index_list_lod[0][i + 1] - 2]
                inference_tmp = logits_tensor[logits_lod[0][i]:logits_lod[0][i +
                                                                             1]]
                labels_tmp = np.array(labels)[logits_lod[0][i]:logits_lod[0][i +
                                                                             1]]

                # some simple post process
                inference_tmp = post_process(inference_tmp)

                # logits -> classification results
                inference_tmp[inference_tmp >= 0.5] = 1
                inference_tmp[inference_tmp < 0.5] = 0
                predict_result = []
                for token in inference_tmp:
                    predict_result.append(np.argwhere(token == 1).tolist())

                # format prediction into spo, calculate metric
                formated_result = format_output(
                    example, predict_result, spo_label_map,
                    tok_to_orig_start_index, tok_to_orig_end_index)
                tp_tmp, fp_tmp, fn_tmp = calculate_metric(
                    example['spo_list'], formated_result['spo_list'])

                tp += tp_tmp
                fp += fp_tmp
                fn += fn_tmp

        except fluid.core.EOFException:
            pyreader.reset()
            break

    time_end = time.time()
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f = 2 * p * r / (p + r) if p + r != 0 else 0
    return "[evaluation] precision: %f, recall: %f, f1: %f, elapsed time: %f s" % (
        p, r, f, time_end - time_begin)


def predict(args, examples, exe, test_program, test_pyreader, graph_vars):

    spo_label_map = json.load(open(args.spo_label_map_config))

    fetch_list = [
        graph_vars["lod_logit"].name, graph_vars["lod_label"].name,
        graph_vars["example_index"].name,
        graph_vars["tok_to_orig_start_index"].name,
        graph_vars["tok_to_orig_end_index"].name
    ]

    test_pyreader.start()
    res = []
    while True:
        try:
            # prepare fetched batch data: unlod etc.
            logits, labels, example_index_list, tok_to_orig_start_index_list, tok_to_orig_end_index_list = \
                exe.run(program=test_program, fetch_list=fetch_list, return_numpy=False)
            example_index_list = np.array(example_index_list).astype(
                int) - 100000
            logits_lod = logits.lod()
            tok_to_orig_start_index_list_lod = tok_to_orig_start_index_list.lod(
            )
            tok_to_orig_end_index_list_lod = tok_to_orig_end_index_list.lod()
            logits_tensor = np.array(logits)
            tok_to_orig_start_index_list = np.array(
                tok_to_orig_start_index_list).flatten()
            tok_to_orig_end_index_list = np.array(
                tok_to_orig_end_index_list).flatten()

            # perform evaluation
            for i in range(len(logits_lod[0]) - 1):
                # prepare prediction results for each example
                example_index = example_index_list[i]
                example = examples[example_index]
                tok_to_orig_start_index = tok_to_orig_start_index_list[
                    tok_to_orig_start_index_list_lod[0][
                        i]:tok_to_orig_start_index_list_lod[0][i + 1] - 2]
                tok_to_orig_end_index = tok_to_orig_end_index_list[
                    tok_to_orig_end_index_list_lod[0][
                        i]:tok_to_orig_end_index_list_lod[0][i + 1] - 2]
                inference_tmp = logits_tensor[logits_lod[0][i]:logits_lod[0][i +
                                                                             1]]

                # some simple post process
                inference_tmp = post_process(inference_tmp)

                # logits -> classification results
                inference_tmp[inference_tmp >= 0.5] = 1
                inference_tmp[inference_tmp < 0.5] = 0
                predict_result = []
                for token in inference_tmp:
                    predict_result.append(np.argwhere(token == 1).tolist())

                # format prediction into spo, calculate metric
                formated_result = format_output(
                    example, predict_result, spo_label_map,
                    tok_to_orig_start_index, tok_to_orig_end_index)

                res.append(formated_result)
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    return res


def post_process(inference):
    # this post process only brings limited improvements (less than 0.5 f1) in order to keep simplicity
    # to obtain better results, CRF is recommended
    reference = []
    for token in inference:
        token_ = token.copy()
        token_[token_ >= 0.5] = 1
        token_[token_ < 0.5] = 0
        reference.append(np.argwhere(token_ == 1))

    #  token was classified into conflict situation (both 'I' and 'B' tag)
    for i, token in enumerate(reference[:-1]):
        if [0] in token and len(token) >= 2:
            if [1] in reference[i + 1]:
                inference[i][0] = 0
            else:
                inference[i][2:] = 0

    #  token wasn't assigned any cls ('B', 'I', 'O' tag all zero)
    for i, token in enumerate(reference[:-1]):
        if len(token) == 0:
            if [1] in reference[i - 1] and [1] in reference[i + 1]:
                inference[i][1] = 1
            elif [1] in reference[i + 1]:
                inference[i][np.argmax(inference[i, 1:]) + 1] = 1

    #  handle with empty spo: to be implemented

    return inference


def format_output(example, predict_result, spo_label_map,
                  tok_to_orig_start_index, tok_to_orig_end_index):
    # format prediction into example-style output
    complex_relation_label = [8, 10, 26, 32, 46]
    complex_relation_affi_label = [9, 11, 27, 28, 29, 33, 47]
    instance = {}
    predict_result = predict_result[1:len(predict_result) -
                                    1]  # remove [CLS] and [SEP]
    text_raw = example['text']

    flatten_predict = []
    for layer_1 in predict_result:
        for layer_2 in layer_1:
            flatten_predict.append(layer_2[0])

    subject_id_list = []
    for cls_label in list(set(flatten_predict)):
        if 1 < cls_label <= 56 and (cls_label + 55) in flatten_predict:
            subject_id_list.append(cls_label)
    subject_id_list = list(set(subject_id_list))

    def find_entity(id_, predict_result):
        entity_list = []
        for i in range(len(predict_result)):
            if [id_] in predict_result[i]:
                j = 0
                while i + j + 1 < len(predict_result):
                    if [1] in predict_result[i + j + 1]:
                        j += 1
                    else:
                        break
                entity = ''.join(text_raw[tok_to_orig_start_index[i]:
                                          tok_to_orig_end_index[i + j] + 1])
                entity_list.append(entity)

        return list(set(entity_list))

    spo_list = []
    for id_ in subject_id_list:
        if id_ in complex_relation_affi_label:
            continue
        if id_ not in complex_relation_label:
            subjects = find_entity(id_, predict_result)
            objects = find_entity(id_ + 55, predict_result)
            for subject_ in subjects:
                for object_ in objects:
                    spo_list.append({
                        "predicate": spo_label_map['predicate'][id_],
                        "object_type": {
                            '@value': spo_label_map['object_type'][id_]
                        },
                        'subject_type': spo_label_map['subject_type'][id_],
                        "object": {
                            '@value': object_
                        },
                        "subject": subject_
                    })
        else:
            #  traverse all complex relation and look through their corresponding affiliated objects
            subjects = find_entity(id_, predict_result)
            objects = find_entity(id_ + 55, predict_result)
            for subject_ in subjects:
                for object_ in objects:
                    object_dict = {'@value': object_}
                    object_type_dict = {
                        '@value':
                        spo_label_map['object_type'][id_].split('_')[0]
                    }

                    if id_ in [8, 10, 32, 46] and id_ + 1 in subject_id_list:
                        id_affi = id_ + 1
                        object_dict[spo_label_map['object_type'][id_affi].split(
                            '_')[1]] = find_entity(id_affi + 55,
                                                   predict_result)[0]
                        object_type_dict[spo_label_map['object_type'][
                            id_affi].split('_')[1]] = spo_label_map[
                                'object_type'][id_affi].split('_')[0]
                    elif id_ == 26:
                        for id_affi in [27, 28, 29]:
                            if id_affi in subject_id_list:
                                object_dict[spo_label_map['object_type'][id_affi].split('_')[1]] = \
                                find_entity(id_affi + 55, predict_result)[0]
                                object_type_dict[spo_label_map['object_type'][id_affi].split('_')[1]] = \
                                spo_label_map['object_type'][id_affi].split('_')[0]

                    spo_list.append({
                        "predicate": spo_label_map['predicate'][id_],
                        "object_type": object_type_dict,
                        "subject_type": spo_label_map['subject_type'][id_],
                        "object": object_dict,
                        "subject": subject_
                    })

    instance['text'] = example['text']
    instance['spo_list'] = spo_list
    return instance
