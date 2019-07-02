# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
This Module provide pretrained word-embeddings 
"""

from __future__ import print_function, unicode_literals
import numpy as np
import time, datetime
import os, sys


def maybe_open(filepath):
    if sys.version_info <= (3, 0):  # for python2
        return open(filepath, 'r')
    else:
        return open(filepath, 'r', encoding="utf-8")


def Glove840B_300D(filepath, keys=None):
    """
    input: the "glove.840B.300d.txt" file path
    return: a dict, key: word (unicode), value: a numpy array with shape [300]
    """
    if keys is not None:
        assert (isinstance(keys, set))
    print("loading word2vec from ", filepath)
    print("please wait for a minute.")
    start = time.time()
    word2vec = {}
    with maybe_open(filepath) as f:
        for line in f:
            if sys.version_info <= (3, 0):  # for python2
                line = line.decode('utf-8')
            info = line.strip("\n").split(" ")
            word = info[0]
            if (keys is not None) and (word not in keys):
                continue
            vector = info[1:]
            assert (len(vector) == 300)
            word2vec[word] = np.asarray(vector, dtype='float32')

    end = time.time()
    print(
        "Spent ",
        str(datetime.timedelta(seconds=end - start)),
        " on loading word2vec.")
    return word2vec


if __name__ == '__main__':
    from os.path import expanduser
    home = expanduser("~")
    embed_dict = Glove840B_300D(
        os.path.join(home, "./.cache/paddle/dataset/glove.840B.300d.txt"))
    exit(0)
