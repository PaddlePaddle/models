#!/usr/bin/env python
# -*- coding: utf-8 -*-
######################################################################
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
######################################################################
"""
File: data_provider.py
"""

import re
import os
import types
import csv
import random
import numpy as np


VOC_DICT = {}


def load_dict(vocab_dict): 
    """
    load vocabulary dict
    """
    idx = 0
    for line in open(vocab_dict): 
        line = line.strip()
        VOC_DICT[line] = idx
        idx += 1
    return VOC_DICT


def prepare_batch_data(insts,
                       task_name,
                       max_len=128,
                       return_attn_bias=True,
                       return_max_len=True,
                       return_num_token=False):
    """
    generate self attention mask, [shape: batch_size *  max_len * max_len]
    """
    batch_context_ids = [inst[0] for inst in insts]
    batch_context_pos_ids = [inst[1] for inst in insts]
    batch_segment_ids = [inst[2] for inst in insts]
    batch_label_ids = [[inst[3]] for inst in insts]
    labels_list = batch_label_ids

    context_id, next_sent_context_index, context_attn_bias = \
            pad_batch_data(batch_context_ids, pad_idx=0, max_len=max_len, \
            return_next_sent_pos=True, return_attn_bias=True)

    context_pos_id = pad_batch_data(
        batch_context_pos_ids, pad_idx=0, max_len=max_len, return_pos=False, return_attn_bias=False)

    context_segment_id = pad_batch_data(
        batch_segment_ids, pad_idx=0, max_len=max_len, return_pos=False, return_attn_bias=False)

    if 'kn' in task_name: 
        batch_kn_ids = [inst[4] for inst in insts]
        kn_id = pad_bath_kn_data(batch_kn_ids, pad_idx=0, max_len=max_len)

    out_list = []
    for i in range(len(insts)): 
        if 'kn' in task_name: 
            out = [context_id[i], context_pos_id[i], context_segment_id[i], context_attn_bias[i], \
                    kn_id[i], labels_list[i], next_sent_context_index[i]]
        else: 
            out = [context_id[i], context_pos_id[i], context_segment_id[i], \
                    context_attn_bias[i], labels_list[i], next_sent_context_index[i]]
        out_list.append(out)
    return out_list


def pad_bath_kn_data(insts,
                    pad_idx=0,
                    max_len=128): 
    kn_list = []
    for inst in insts: 
        inst = inst[0: min(max_len, len(inst))]
        kn_list.append(inst)
    return kn_list


def pad_batch_data(insts,
                   pad_idx=0,
                   max_len=128,
                   return_pos=False,
                   return_next_sent_pos=False,
                   return_attn_bias=False,
                   return_max_len=False,
                   return_num_token=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    if return_next_sent_pos:
        batch_size = inst_data.shape[0]
        max_seq_len = inst_data.shape[1]
        next_sent_index = np.array(
            range(0, batch_size * max_seq_len, max_seq_len)).astype(
                "int64").reshape(-1, 1)
        return_list += [next_sent_index]

    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts])
        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_attn_bias:
        slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                       (max_len - len(inst)) for inst in insts])
        slf_attn_bias_data = np.tile(
            slf_attn_bias_data.reshape([-1, 1, max_len]), [1, max_len, 1])
        return_list += [slf_attn_bias_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    return return_list if len(return_list) > 1 else return_list[0]


def preprocessing_for_one_line(line, labels, task_name, max_seq_len=256):
    """
    process text to model inputs
    """
    line = line.rstrip('\n').split('\t')
    label_text = line[0]
    context_text = line[1]
    response_text = line[2]
    if 'kn' in task_name:
        kn_text = "%s [SEP] %s" % (line[3], line[4])
    else:
        kn_text = None

    example = InputExample(guid=0, \
                           context_text=context_text, \
                           response_text=response_text, \
                           kn_text=kn_text, \
                           label_text=label_text)

    feature = convert_single_example(0, example, labels, max_seq_len)

    instance = [feature.context_ids, feature.context_pos_ids, \
                feature.segment_ids, feature.label_ids, feature.kn_ids]

    batch_data = prepare_batch_data([instance],
                                    task_name,
                                    max_len=max_seq_len,
                                    return_attn_bias=True,
                                    return_max_len=False,
                                    return_num_token=False)

    return batch_data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, data_dir, task_name, vocab_path, max_seq_len, do_lower_case):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.task_name = task_name

        self.current_train_example = -1
        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}
        self.current_train_epoch = -1
        VOC_DICT = load_dict(vocab_path)

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    @classmethod
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def convert_example(self, index, example, labels, max_seq_len):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        feature = convert_single_example(index, example, labels, max_seq_len)
        return feature

    def generate_batch_data(self,
                            batch_data,
                            voc_size=-1,
                            mask_id=-1,
                            return_attn_bias=True,
                            return_max_len=False,
                            return_num_token=False):
        return prepare_batch_data(
            batch_data,
            self.task_name,
            self.max_seq_len,
            return_attn_bias=True,
            return_max_len=False,
            return_num_token=False)

    @classmethod
    def _read_data(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            lines = []
            for line in f:
                line = line.rstrip('\n').split('\t')
                lines.append(line)
            return lines

    def get_num_examples(self, phase):
        """Get number of examples for train, dev or test."""
        if phase not in ['train', 'dev', 'test']:
            raise ValueError("Unknown phase, which should be in ['train', 'dev', 'test'].")
        return self.num_examples[phase]

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_train_example, self.current_train_epoch

    def data_generator(self,
                       batch_size,
                       phase='train',
                       epoch=1,
                       shuffle=False):
        """
        Generate data for train, dev or test.
        """
        if phase == 'train':
            examples = self.get_train_examples(self.data_dir)
            self.num_examples['train'] = len(examples)
        elif phase == 'dev':
            examples = self.get_dev_examples(self.data_dir)
            self.num_examples['dev'] = len(examples)
        elif phase == 'test':
            examples = self.get_test_examples(self.data_dir)
            self.num_examples['test'] = len(examples)
        else:
            raise ValueError("Unknown phase, which should be in ['train', 'dev', 'test'].")

        def instance_reader():
            for epoch_index in range(epoch):
                if shuffle:
                    random.shuffle(examples)
                if phase == 'train':
                    self.current_train_epoch = epoch_index
                for (index, example) in enumerate(examples):
                    if phase == 'train':
                        self.current_train_example = index + 1
                    feature = self.convert_example(
                        index, example, self.get_labels(), self.max_seq_len)
                    if 'kn' in self.task_name: 
                        instance = [feature.context_ids, feature.context_pos_ids, \
                                feature.segment_ids, feature.label_ids, feature.kn_ids]
                    else: 
                        instance = [feature.context_ids, feature.context_pos_ids, \
                                feature.segment_ids, feature.label_ids]
                    yield instance

        def batch_reader(reader, batch_size):
            batch = []
            for instance in reader():
                if len(batch) < batch_size: 
                    batch.append(instance)
                else: 
                    yield batch
                    batch = [instance]

            if len(batch) > 0:
                yield batch

        def wrapper():
            for batch_data in batch_reader(instance_reader, batch_size):
                batch_data = self.generate_batch_data(
                    batch_data,
                    voc_size=-1,
                    mask_id=-1,
                    return_attn_bias=True,
                    return_max_len=False,
                    return_num_token=False)
                yield batch_data

        return wrapper


class InputExample(object):
    """A single training/test example"""

    def __init__(self, guid, context_text, response_text, kn_text, label_text):
        self.guid = guid
        self.context_text = context_text
        self.response_text = response_text
        self.kn_text = kn_text
        self.label_text = label_text


class InputFeatures(object): 
    """input features datas"""
    def __init__(self, context_ids, context_pos_ids, segment_ids, kn_ids, label_ids): 
        self.context_ids = context_ids
        self.context_pos_ids = context_pos_ids
        self.segment_ids = segment_ids
        self.kn_ids = kn_ids
        self.label_ids = label_ids


class MatchProcessor(DataProcessor):
    """Processor for the Match data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    @classmethod
    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            context_text = line[1]
            label_text = line[0]
            response_text = line[2]
            if 'kn' in self.task_name: 
                kn_text = "%s [SEP] %s" % (line[3], line[4])
            else: 
                kn_text = None
            examples.append(
                InputExample(
                    guid=guid, context_text=context_text, response_text=response_text, \
                            kn_text=kn_text, label_text=label_text))
        return examples


def convert_tokens_to_ids(tokens): 
    """
    convert input ids
    """
    ids = []
    for token in tokens: 
        if token in VOC_DICT: 
            ids.append(VOC_DICT[token])
        else: 
            ids.append(VOC_DICT['[UNK]'])
    return ids


def convert_single_example(ex_index, example, label_list, max_seq_length):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    if example.context_text: 
        tokens_context = example.context_text
        tokens_context = tokens_context.split()
    else: 
        tokens_context = []

    if example.response_text: 
        tokens_response = example.response_text
        tokens_response = tokens_response.split()
    else: 
        tokens_response = []

    if example.kn_text:
        tokens_kn = example.kn_text
        tokens_kn = tokens_kn.split()
        tokens_kn = tokens_kn[0: min(len(tokens_kn), max_seq_length)]
    else: 
        tokens_kn = []

    tokens_response = tokens_response[0: min(50, len(tokens_response))]
    if len(tokens_context) > max_seq_length - len(tokens_response) - 3: 
        tokens_context = tokens_context[len(tokens_context) \
                + len(tokens_response) - max_seq_length + 3:]

    context_tokens = []
    segment_ids = []

    context_tokens.append("[CLS]")
    segment_ids.append(0)
    context_tokens.extend(tokens_context)
    segment_ids.extend([0] * len(tokens_context))
    context_tokens.append("[SEP]")
    segment_ids.append(0)

    context_tokens.extend(tokens_response)
    segment_ids.extend([1] * len(tokens_response))
    context_tokens.append("[SEP]")
    segment_ids.append(1)

    context_ids = convert_tokens_to_ids(context_tokens)
    context_pos_ids = list(range(len(context_ids)))
    label_ids = label_map[example.label_text]
    if tokens_kn: 
        kn_ids = convert_tokens_to_ids(tokens_kn)
    else: 
        kn_ids = []

    feature = InputFeatures(
        context_ids=context_ids,
        context_pos_ids=context_pos_ids,
        segment_ids=segment_ids,
        kn_ids = kn_ids,
        label_ids=label_ids)
    #if ex_index < 5:
    #    print("*** Example ***")
    #    print("guid: %s" % (example.guid))
    #    print("context tokens: %s" % " ".join(context_tokens))
    #    print("context_ids: %s" % " ".join([str(x) for x in context_ids]))
    #    print("context_pos_ids: %s" % " ".join([str(x) for x in context_pos_ids]))
    #    print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #    print("kn_ids: %s" % " ".join([str(x) for x in kn_ids]))
    #    print("label: %s (id = %d)" % (example.label_text, label_ids))
    return feature

