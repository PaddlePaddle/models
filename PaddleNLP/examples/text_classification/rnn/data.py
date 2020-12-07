# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from collections import namedtuple
import csv
import io
import os

import paddle


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n").split("\t")[0]
        vocab[token] = index
    return vocab


def convert_tokens_to_ids(tokens, vocab):
    """ Converts a token string (or a sequence of tokens) in a single integer id
        (or a sequence of ids), using the vocabulary.
    """

    ids = []
    for token in tokens:
        wid = vocab.get(token, None)
        if wid:
            ids.append(wid)
    return ids


class ChnSentiCorp(paddle.io.Dataset):
    """
    ChnSentiCorp is a dataset for chinese sentiment classification,
    which was published by Tan Songbo at ICT of Chinese Academy of Sciences.

    Args:
        base_path (:obj:`str`) : The dataset file path, which contains train.tsv, dev.tsv and test.tsv.
        mode (:obj:`str`, `optional`, defaults to `train`):
            It identifies the dataset mode (train, test or dev).
    """

    def __init__(self, base_path, vocab, mode='train'):
        if mode == 'train':
            data_file = 'train.tsv'
        elif mode == 'test':
            data_file = 'test.tsv'
        else:
            data_file = 'dev.tsv'

        self.data_file = os.path.join(base_path, data_file)
        self.label_list = ["0", "1"]
        self.label_map = {
            item: index
            for index, item in enumerate(self.label_list)
        }
        self.vocab = vocab

        self.raw_examples = self._read_file(self.data_file)

    def _read_file(self, input_file):
        """
        Reads a tab separated value file.

        Args:
            input_file (:obj:`str`) : The file to be read.

        Returns:
            examples (:obj:`list`): All the input data.
        """
        if not os.path.exists(input_file):
            raise RuntimeError("The file {} is not found.".format(input_file))
        else:
            Example = namedtuple('Example', ['text', 'label', 'seq_len'])
            with io.open(input_file, "r", encoding="UTF-8") as f:
                reader = csv.reader(f, delimiter="\t", quotechar=None)
                examples = []
                header = next(reader)
                for line in reader:
                    tokens = line[0].strip().split(' ')
                    ids = convert_tokens_to_ids(tokens, self.vocab)
                    example = Example(
                        text=ids,
                        label=self.label_map[line[1]],
                        seq_len=len(ids))
                    examples.append(example)
                return examples

    def __getitem__(self, idx):
        return self.raw_examples[idx]

    def __len__(self):
        return len(self.raw_examples)


if __name__ == "__main__":
    vocab = load_vocab('./senta_data/word_dict.txt')
    train_dataset = ChnSentiCorp(
        base_path='./senta_data', vocab=vocab, mode='train')
    dev_dataset = ChnSentiCorp(
        base_path='./senta_data', vocab=vocab, mode='dev')
    test_dataset = ChnSentiCorp(
        base_path='./senta_data', vocab=vocab, mode='test')
    index = 0
    for example in train_dataset:
        print("%s \t %s \t %s" % (example.text, example.label, example.seq_len))
        index += 1
        if index > 3:
            break
