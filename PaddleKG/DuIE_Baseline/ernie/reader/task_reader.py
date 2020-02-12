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

import sys
import os
import json
import random
import logging
import numpy as np
import six
from io import open
from collections import namedtuple

import tokenization
from batching import pad_batch_data

import extract_chinese_and_punct

log = logging.getLogger(__name__)

if six.PY3:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class BaseReader(object):
    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 is_inference=False,
                 random_seed=None,
                 tokenizer="FullTokenizer",
                 is_classify=True,
                 is_regression=False,
                 for_cn=True,
                 task_id=0):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = in_tokens
        self.is_inference = is_inference
        self.for_cn = for_cn
        self.task_id = task_id

        np.random.seed(random_seed)

        self.is_classify = is_classify
        self.is_regression = is_regression
        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        if label_map_config:
            with open(label_map_config, encoding='utf8') as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


class RelationExtractionMultiCLSReader(BaseReader):
    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 spo_label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 is_inference=False,
                 random_seed=None,
                 tokenizer="FullTokenizer",
                 is_classify=True,
                 is_regression=False,
                 for_cn=True,
                 task_id=0,
                 num_labels=0):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.chineseandpunctuationextractor = extract_chinese_and_punct.ChineseAndPunctuationExtractor(
        )
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = in_tokens
        self.is_inference = is_inference
        self.for_cn = for_cn
        self.task_id = task_id
        self.num_labels = num_labels

        np.random.seed(random_seed)

        self.is_classify = is_classify
        self.is_regression = is_regression
        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0
        # map string to relation id
        if label_map_config:
            with open(label_map_config, encoding='utf8') as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None
        # map relation id to string(including subject name, predicate name, object name)
        if spo_label_map_config:
            with open(label_map_config, encoding='utf8') as f:
                self.label_map = json.load(f)
        else:
            self.spo_label_map = None

    def _read_json(self, input_file):
        f = open(input_file, 'r', encoding="utf8")
        examples = []
        for line in f.readlines():
            examples.append(json.loads(line))
        f.close()
        return examples

    def get_num_examples(self, input_file):
        examples = self._read_json(input_file)
        return len(examples)

    def _prepare_batch_data(self, examples, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example = index
            example_index = 100000 + index
            record = self._convert_example_to_record(
                example_index, example, self.max_seq_len, self.tokenizer)
            max_len = max(max_len, len(record.token_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self._pad_batch_records(batch_records)
                batch_records, max_len = [record], len(record.token_ids)

        if batch_records:
            yield self._pad_batch_records(batch_records)

    def data_generator(self,
                       input_file,
                       batch_size,
                       epoch,
                       dev_count=1,
                       shuffle=True,
                       phase=None):
        examples = self._read_json(input_file)

        def wrapper():
            all_dev_batches = []
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                if shuffle:
                    np.random.shuffle(examples)

                for batch_data in self._prepare_batch_data(
                        examples, batch_size, phase=phase):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        for batch in all_dev_batches:
                            yield batch
                        all_dev_batches = []

        def f():
            try:
                for i in wrapper():
                    yield i
            except Exception as e:
                import traceback
                traceback.print_exc()

        return f

    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_label_ids = [record.label_ids for record in batch_records]
        batch_example_index = [record.example_index for record in batch_records]
        batch_tok_to_orig_start_index = [
            record.tok_to_orig_start_index for record in batch_records
        ]
        batch_tok_to_orig_end_index = [
            record.tok_to_orig_end_index for record in batch_records
        ]
        # padding
        padded_token_ids, input_mask, batch_seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)

        #  label padding for expended dimension
        outside_label = np.array([1] + [0] * (self.num_labels - 1))
        max_len = max(len(inst) for inst in batch_label_ids)
        padded_label_ids = []
        for i, inst in enumerate(batch_label_ids):
            inst = np.concatenate(
                (np.array(inst), np.tile(outside_label, ((max_len - len(inst)),
                                                         1))),
                axis=0)
            padded_label_ids.append(inst)
        padded_label_ids = np.stack(padded_label_ids).astype("float32")

        padded_tok_to_orig_start_index = np.array([
            inst + [0] * (max_len - len(inst))
            for inst in batch_tok_to_orig_start_index
        ])
        padded_tok_to_orig_end_index = np.array([
            inst + [0] * (max_len - len(inst))
            for inst in batch_tok_to_orig_end_index
        ])

        padded_task_ids = np.ones_like(
            padded_token_ids, dtype="int64") * self.task_id

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            padded_task_ids, input_mask, padded_label_ids, batch_seq_lens,
            batch_example_index, padded_tok_to_orig_start_index,
            padded_tok_to_orig_end_index
        ]
        return return_list

    def _convert_example_to_record(self, example_index, example, max_seq_length,
                                   tokenizer):
        spo_list = example['spo_list']
        text_raw = example['text']

        sub_text = []
        buff = ""
        for char in text_raw:
            if self.chineseandpunctuationextractor.is_chinese_or_punct(char):
                if buff != "":
                    sub_text.append(buff)
                    buff = ""
                sub_text.append(char)
            else:
                buff += char
        if buff != "":
            sub_text.append(buff)

        tok_to_orig_start_index = []
        tok_to_orig_end_index = []
        orig_to_tok_index = []
        tokens = []
        text_tmp = ''
        for (i, token) in enumerate(sub_text):
            orig_to_tok_index.append(len(tokens))
            sub_tokens = tokenizer.tokenize(token)
            text_tmp += token
            for sub_token in sub_tokens:
                tok_to_orig_start_index.append(len(text_tmp) - len(token))
                tok_to_orig_end_index.append(len(text_tmp) - 1)
                tokens.append(sub_token)
                if len(tokens) >= max_seq_length - 2:
                    break
            else:
                continue
            break

        labels = [[0] * self.num_labels
                  for i in range(len(tokens))]  # initialize tag
        #  find all entities and tag them with corresponding "B"/"I" labels
        for spo in spo_list:
            for spo_object in spo['object'].keys():
                # assign relation label
                if spo['predicate'] in self.label_map.keys():
                    # simple relation
                    label_subject = self.label_map[spo['predicate']]
                    label_object = label_subject + 55
                    subject_sub_tokens = tokenizer.tokenize(spo['subject'])
                    object_sub_tokens = tokenizer.tokenize(spo['object'][
                        '@value'])
                else:
                    # complex relation
                    label_subject = self.label_map[spo['predicate'] + '_' +
                                                   spo_object]
                    label_object = label_subject + 55
                    subject_sub_tokens = tokenizer.tokenize(spo['subject'])
                    object_sub_tokens = tokenizer.tokenize(spo['object'][
                        spo_object])

                # assign token label
                # there are situations where s entity and o entity might overlap, e.g. xyz established xyz corporation
                # to prevent single token from being labeled into two different entity
                # we tag the longer entity first, then match the shorter entity within the rest text
                forbidden_index = None
                if len(subject_sub_tokens) > len(object_sub_tokens):
                    for index in range(
                            len(tokens) - len(subject_sub_tokens) + 1):
                        if tokens[index:index + len(
                                subject_sub_tokens)] == subject_sub_tokens:
                            labels[index][label_subject] = 1
                            for i in range(len(subject_sub_tokens) - 1):
                                labels[index + i + 1][1] = 1
                            forbidden_index = index
                            break

                    for index in range(
                            len(tokens) - len(object_sub_tokens) + 1):
                        if tokens[index:index + len(
                                object_sub_tokens)] == object_sub_tokens:
                            if forbidden_index is None:
                                labels[index][label_object] = 1
                                for i in range(len(object_sub_tokens) - 1):
                                    labels[index + i + 1][1] = 1
                                break
                            # check if labeled already
                            elif index < forbidden_index or index >= forbidden_index + len(
                                    subject_sub_tokens):
                                labels[index][label_object] = 1
                                for i in range(len(object_sub_tokens) - 1):
                                    labels[index + i + 1][1] = 1
                                break

                else:
                    for index in range(
                            len(tokens) - len(object_sub_tokens) + 1):
                        if tokens[index:index + len(
                                object_sub_tokens)] == object_sub_tokens:
                            labels[index][label_object] = 1
                            for i in range(len(object_sub_tokens) - 1):
                                labels[index + i + 1][1] = 1
                            forbidden_index = index
                            break

                    for index in range(
                            len(tokens) - len(subject_sub_tokens) + 1):
                        if tokens[index:index + len(
                                subject_sub_tokens)] == subject_sub_tokens:
                            if forbidden_index is None:
                                labels[index][label_subject] = 1
                                for i in range(len(subject_sub_tokens) - 1):
                                    labels[index + i + 1][1] = 1
                                break
                            elif index < forbidden_index or index >= forbidden_index + len(
                                    object_sub_tokens):
                                labels[index][label_subject] = 1
                                for i in range(len(subject_sub_tokens) - 1):
                                    labels[index + i + 1][1] = 1
                                break

        # if token wasn't assigned as any "B"/"I" tag, give it an "O" tag for outside
        for i in range(len(labels)):
            if labels[i] == [0] * self.num_labels:
                labels[i][0] = 1

        # add [CLS] and [SEP] token, they are tagged into "O" for outside
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        outside_label = [[1] + [0] * (self.num_labels - 1)]
        labels = outside_label + labels + outside_label

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))
        text_type_ids = [0] * len(token_ids)

        Record = namedtuple('Record', [
            'token_ids', 'text_type_ids', 'position_ids', 'label_ids',
            'example_index', 'tok_to_orig_start_index', 'tok_to_orig_end_index'
        ])
        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
            position_ids=position_ids,
            label_ids=labels,
            example_index=example_index,
            tok_to_orig_start_index=tok_to_orig_start_index,
            tok_to_orig_end_index=tok_to_orig_end_index)
        return record


if __name__ == '__main__':
    pass
