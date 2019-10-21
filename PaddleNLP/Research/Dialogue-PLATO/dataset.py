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
"""
Dataset class
"""

import json


class Dataset(object):
    """ Basic Dataset interface class. """

    @classmethod
    def add_cmdline_argument(cls, parser):
        group = parser.add_argument_group("Dataset")
        group.add_argument("--data_dir", type=str, required=True,
                           help="The dataset dir.")
        group.add_argument("--data_type", type=str, required=True,
                           choices=["multi", "multi_knowledge"],
                           help="The type of dataset.")
        return group

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LazyDataset(Dataset):
    """
    Lazy load dataset from disk.

    Each line of data file is a preprocessed example.
    """

    def __init__(self, data_file, transform=lambda s: json.loads(s)):
        """
        Initialize lazy dataset.

        By default, loading .jsonl format.

        :param data_file
        :type str

        :param transform
        :type callable
        """
        self.data_file = data_file
        self.transform = transform
        self.offsets = [0]
        with open(data_file, "r", encoding="utf-8") as fp:
            while fp.readline() != "":
                self.offsets.append(fp.tell())
        self.offsets.pop()
        self.fp = open(data_file, "r", encoding="utf-8")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        self.fp.seek(self.offsets[idx], 0)
        return self.transform(self.fp.readline().strip())
