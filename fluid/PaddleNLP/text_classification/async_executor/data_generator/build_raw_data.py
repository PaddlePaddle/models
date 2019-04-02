#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
Build raw data
"""
from __future__ import print_function
import sys
import os
import random
import re
data_type = sys.argv[1]

if not (data_type == "train" or data_type == "test"):
    print("python %s [test/train]" % sys.argv[0], file=sys.stderr)
    sys.exit(-1)

pos_folder = "aclImdb/" + data_type + "/pos/"
neg_folder = "aclImdb/" + data_type + "/neg/"

pos_train_list = [(pos_folder + x, "1") for x in os.listdir(pos_folder)]
neg_train_list = [(neg_folder + x, "0") for x in os.listdir(neg_folder)]

all_train_list = pos_train_list + neg_train_list
random.shuffle(all_train_list)


def load_dict(dictfile):
    """
    Load word id dict
    """
    vocab = {}
    wid = 0
    with open(dictfile) as f:
        for line in f:
            vocab[line.strip()] = str(wid)
            wid += 1
    return vocab


vocab = load_dict("aclImdb/imdb.vocab")
unk_id = str(len(vocab))
print("vocab size: ", len(vocab), file=sys.stderr)
pattern = re.compile(r'(;|,|\.|\?|!|\s|\(|\))')

for fitem in all_train_list:
    label = str(fitem[1])
    fname = fitem[0]
    with open(fname) as f:
        sent = f.readline().lower().replace("<br />", " ").strip()
        out_s = "%s | %s" % (sent, label)
        print(out_s, file=sys.stdout)
