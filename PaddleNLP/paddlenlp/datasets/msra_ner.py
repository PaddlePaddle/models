# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import os
import warnings

from paddle.io import Dataset
from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from paddlenlp.datasets import DatasetReader

from .dataset import TSVDataset

__all__ = ['MSRA_NER', 'MSRA_NER_new', 'MSRA_NER_new_iter']


class MSRA_NER_new(DatasetReader):
    URL = "https://paddlenlp.bj.bcebos.com/datasets/msra_ner.tar.gz"
    MD5 = None
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(os.path.join('msra_ner', 'train.tsv'), None),
        'test': META_INFO(os.path.join('msra_ner', 'test.tsv'), None)
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):

            get_path_from_url(self.URL, default_root, self.MD5)
            fullname = os.path.join(default_root, filename)

        return fullname

    def _read(self, filename):
        examples = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line_stripped = line.strip().split('\t')
                if not line_stripped:
                    break
                if len(line_stripped) == 2:
                    tokens = line_stripped[0].split("\002")
                    tags = line_stripped[1].split("\002")
                else:
                    tokens = line_stripped.split("\002")
                    tags = []
                examples.append({"tokens": tokens, "labels": tags})

        return examples

    def get_labels(self):

        return ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]


class MSRA_NER_new_iter(DatasetReader):
    URL = "https://paddlenlp.bj.bcebos.com/datasets/msra_ner.tar.gz"
    MD5 = None
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(os.path.join('msra_ner', 'train.tsv'), None),
        'test': META_INFO(os.path.join('msra_ner', 'test.tsv'), None)
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):

            get_path_from_url(self.URL, default_root, self.MD5)
            fullname = os.path.join(default_root, filename)

        return fullname

    def _read(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line_stripped = line.strip().split('\t')
                if not line_stripped:
                    break
                if len(line_stripped) == 2:
                    tokens = line_stripped[0].split("\002")
                    tags = line_stripped[1].split("\002")
                else:
                    tokens = line_stripped.split("\002")
                    tags = []
                yield {"tokens": tokens, "labels": tags}

    def get_labels(self):

        return ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]


class MSRA_NER(TSVDataset):
    URL = "https://paddlenlp.bj.bcebos.com/datasets/msra_ner.tar.gz"
    MD5 = None
    META_INFO = collections.namedtuple(
        'META_INFO', ('file', 'md5', 'field_indices', 'num_discard_samples'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('msra_ner', 'train.tsv'),
            '67d3c93a37daba60ef43c03271f119d7',
            (0, 1),
            0, ),
        'test': META_INFO(
            os.path.join('msra_ner', 'test.tsv'),
            '2f27ae68b5f61d6553ffa28bb577c8a7',
            (0, 1),
            0, ),
    }

    def __init__(self, mode='train', root=None, **kwargs):
        default_root = os.path.join(DATA_HOME, 'msra')
        filename, data_hash, field_indices, num_discard_samples = self.SPLITS[
            mode]
        fullname = os.path.join(default_root,
                                filename) if root is None else os.path.join(
                                    os.path.expanduser(root), filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            if root is not None:  # not specified, and no need to warn
                warnings.warn(
                    'md5 check failed for {}, download {} data to {}'.format(
                        filename, self.__class__.__name__, default_root))
            path = get_path_from_url(self.URL, default_root, self.MD5)
            fullname = os.path.join(default_root, filename)
        super(MSRA_NER_old, self).__init__(
            fullname,
            field_indices=field_indices,
            num_discard_samples=num_discard_samples,
            **kwargs)

    def get_labels(self):
        """
        Return labels of the GlueCoLA object.
        """
        return ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]
