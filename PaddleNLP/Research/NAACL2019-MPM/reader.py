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
""" module for data reader """

import os
import sys
import re
import types
import csv
import random
import numpy as np

from batching import prepare_batch_data

sys.path.append('./BERT')
import tokenization


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self,
                 data_dir,
                 vocab_path,
                 max_seq_len,
                 do_lower_case,
                 in_tokens,
                 random_seed=None):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.in_tokens = in_tokens

        np.random.seed(random_seed)

        self.current_train_example = -1
        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}
        self.current_train_epoch = -1

    def get_train_examples(self, data_dir, drop_keyword):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def convert_example(self, index, example, labels, max_seq_len, tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        feature = convert_single_example(index, example, labels, max_seq_len,
                                         tokenizer)
        return feature

    def generate_instance(self, feature):
        """
        generate instance with given feature

        Args:
            feature: InputFeatures(object). A single set of features of data.
        """
        input_pos = list(range(len(feature.input_ids)))
        return [
            feature.input_ids, feature.segment_ids, input_pos, feature.label_id
        ]

    def generate_batch_data(self,
                            batch_data,
                            total_token_num,
                            voc_size=-1,
                            mask_id=-1,
                            return_input_mask=True,
                            return_max_len=False,
                            return_num_token=False):
        """Generate batch data."""
        return prepare_batch_data(
            batch_data,
            total_token_num,
            voc_size=-1,
            pad_id=self.vocab["[PAD]"],
            cls_id=self.vocab["[CLS]"],
            sep_id=self.vocab["[SEP]"],
            mask_id=-1,
            return_input_mask=True,
            return_max_len=False,
            return_num_token=False)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def get_num_examples(self, phase):
        """Get number of examples for train, dev or test."""
        if phase not in ['train', 'dev', 'test']:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'test'].")
        return self.num_examples[phase]

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_train_example, self.current_train_epoch

    def data_generator_for_kfold(self,
                                 examples,
                                 batch_size,
                                 phase='train',
                                 epoch=1,
                                 dev_count=1,
                                 shuffle=True):
        """
        Generate data for train, dev or test.
    
        Args:
          examples: list. Train, dev or test data.
          batch_size: int. The batch size of generated data.
          phase: string. The phase for which to generate data.
          epoch: int. Total epoches to generate data.
          shuffle: bool. Whether to shuffle examples.
        """
        if phase == 'train':
            self.num_examples['train'] = len(examples)
        elif phase == 'dev':
            self.num_examples['dev'] = len(examples)
        elif phase == 'test':
            self.num_examples['test'] = len(examples)
        else:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'test'].")

        def instance_reader():
            """Process sinle example and return."""
            for epoch_index in range(epoch):
                if shuffle:
                    np.random.shuffle(examples)
                if phase == 'train':
                    self.current_train_epoch = epoch_index
                for (index, example) in enumerate(examples):
                    if phase == 'train':
                        self.current_train_example = index + 1
                    feature = self.convert_example(
                        index, example,
                        self.get_labels(), self.max_seq_len, self.tokenizer)

                    instance = self.generate_instance(feature)
                    yield instance

        def batch_reader(reader, batch_size, in_tokens):
            """Generate batch data and return."""
            batch, total_token_num, max_len = [], 0, 0
            for instance in reader():
                token_ids, sent_ids, pos_ids, label = instance[:4]
                max_len = max(max_len, len(token_ids))
                if in_tokens:
                    to_append = (len(batch) + 1) * max_len <= batch_size
                else:
                    to_append = len(batch) < batch_size
                if to_append:
                    batch.append(instance)
                    total_token_num += len(token_ids)
                else:
                    yield batch, total_token_num
                    batch, total_token_num, max_len = [instance], len(
                        token_ids), len(token_ids)

            if len(batch) > 0:
                yield batch, total_token_num

        def wrapper():
            """Data wrapeer."""
            all_dev_batches = []
            for batch_data, total_token_num in batch_reader(
                    instance_reader, batch_size, self.in_tokens):
                batch_data = self.generate_batch_data(
                    batch_data,
                    total_token_num,
                    voc_size=-1,
                    mask_id=-1,
                    return_input_mask=True,
                    return_max_len=False,
                    return_num_token=False)
                if len(all_dev_batches) < dev_count:
                    all_dev_batches.append(batch_data)

                if len(all_dev_batches) == dev_count:
                    for batch in all_dev_batches:
                        yield batch
                    all_dev_batches = []

        return wrapper

    def data_generator(self,
                       batch_size,
                       phase='train',
                       epoch=1,
                       dev_count=1,
                       shuffle=True,
                       drop_keyword=False):
        """
        Generate data for train, dev or test.
    
        Args:
          batch_size: int. The batch size of generated data.
          phase: string. The phase for which to generate data.
          epoch: int. Total epoches to generate data.
          shuffle: bool. Whether to shuffle examples.
        """
        if phase == 'train':
            examples = self.get_train_examples(
                self.data_dir, drop_keyword=drop_keyword)
            self.num_examples['train'] = len(examples)
        elif phase == 'dev':
            examples = self.get_dev_examples(self.data_dir)
            self.num_examples['dev'] = len(examples)
        elif phase == 'test':
            examples = self.get_test_examples(self.data_dir)
            self.num_examples['test'] = len(examples)
        else:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'test'].")

        def instance_reader():
            """Process sinle example and return."""
            for epoch_index in range(epoch):
                if shuffle:
                    np.random.shuffle(examples)
                if phase == 'train':
                    self.current_train_epoch = epoch_index
                for (index, example) in enumerate(examples):
                    if phase == 'train':
                        self.current_train_example = index + 1
                    feature = self.convert_example(
                        index, example,
                        self.get_labels(), self.max_seq_len, self.tokenizer)

                    instance = self.generate_instance(feature)
                    yield instance

        def batch_reader(reader, batch_size, in_tokens):
            """Generate batch data and return."""
            batch, total_token_num, max_len = [], 0, 0
            for instance in reader():
                token_ids, sent_ids, pos_ids, label = instance[:4]
                max_len = max(max_len, len(token_ids))
                if in_tokens:
                    to_append = (len(batch) + 1) * max_len <= batch_size
                else:
                    to_append = len(batch) < batch_size
                if to_append:
                    batch.append(instance)
                    total_token_num += len(token_ids)
                else:
                    yield batch, total_token_num
                    batch, total_token_num, max_len = [instance], len(
                        token_ids), len(token_ids)

            if len(batch) > 0:
                yield batch, total_token_num

        def wrapper():
            """Data wrapeer."""
            all_dev_batches = []
            for batch_data, total_token_num in batch_reader(
                    instance_reader, batch_size, self.in_tokens):
                batch_data = self.generate_batch_data(
                    batch_data,
                    total_token_num,
                    voc_size=-1,
                    mask_id=-1,
                    return_input_mask=True,
                    return_max_len=False,
                    return_num_token=False)
                if len(all_dev_batches) < dev_count:
                    all_dev_batches.append(batch_data)

                if len(all_dev_batches) == dev_count:
                    for batch in all_dev_batches:
                        yield batch
                    all_dev_batches = []

        return wrapper


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class SemevalTask9Processor(DataProcessor):
    """Processor for Semeval Task9 data set."""

    def get_train_examples(self, data_dir, header=False, drop_keyword=False):
        lines = self._read_csv(data_dir + '/V1.4_Training.csv')
        examples = []
        if drop_keyword:
            keywords = [
                line.strip() for line in open(data_dir + '/../keywords')
            ]

        for i, line in enumerate(lines):
            if i == 0 and header:
                continue
            guid = line[0]
            text_a = tokenization.convert_to_unicode(line[1])
            text_a = clean_str(text_a)

            if drop_keyword:
                new_tokens = []
                for w in text_a.split(' '):
                    if w in keywords and random.random() > 0.8:
                        continue
                    new_tokens.append(w)
            text_a = ' '.join(new_tokens)
            text_b = None
            label = line[2]
            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_dev_examples(self, data_dir, header=True):
        lines = self._read_csv(data_dir + '/SubtaskA_Trial_Test_Labeled.csv')
        examples = []
        for i, line in enumerate(lines):
            if i == 0 and header:
                continue
            guid = line[0]
            text_a = clean_str(line[1])
            text_a = tokenization.convert_to_unicode(text_a)
            text_b = None
            label = line[2]
            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir, header=False):
        lines = self._read_csv(data_dir +
                               '/SubtaskA_EvaluationData_labeled.csv')
        examples = []
        for i, line in enumerate(lines):
            if i == 0 and header:
                continue
            guid = line[0]
            text_a = clean_str(line[1])
            text_a = tokenization.convert_to_unicode(text_a)
            text_b = None
            label = line[2]
            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    @classmethod
    def _read_csv(cls, input_file):
        """Reads a comma separated value file."""
        readers = csv.reader(open(input_file, "r"), delimiter=',')
        lines = []
        for line in readers:
            lines.append(line)
        return lines


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py	
    """
    string = string.strip('\n').replace('\n', ' ')
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string


def convert_single_example_to_unicode(guid, single_example):
    """Convert single example to unicode."""
    text_a = tokenization.convert_to_unicode(single_example[0])
    text_b = tokenization.convert_to_unicode(single_example[1])
    label = tokenization.convert_to_unicode(single_example[2])
    return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.

    input_mask = [1] * len(input_ids)

    label_id = label_map[example.label]

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)
    return feature


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


if __name__ == '__main__':
    pass
