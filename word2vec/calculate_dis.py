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
Example:
    python calculate_dis.py DICTIONARYTXT FEATURETXT
Required arguments:
    DICTIONARYTXT    the dictionary generated in dataprovider
    FEATURETXT       the text format word feature, one line for one word
"""

import numpy as np
from argparse import ArgumentParser


def load_dict(fdict):
    words = [line.strip() for line in fdict.readlines()]
    dictionary = dict(zip(words, xrange(len(words))))
    return dictionary


def load_emb(femb):
    feaBank = []
    flag_firstline = True
    for line in femb:
        if flag_firstline:
            flag_firstline = False
            continue
        fea = np.array([float(x) for x in line.strip().split(',')])
        normfea = fea * 1.0 / np.linalg.norm(fea)
        feaBank.append(normfea)
    return feaBank


def calcos(id1, id2, Fea):
    f1 = Fea[id1]
    f2 = Fea[id2]
    return np.dot(f1.transpose(), f2)


def get_wordidx(w, Dict):
    if w not in Dict:
        print 'ERROR: %s not in the dictionary' % w
        return -1
    return Dict[w]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dict', help='dictionary file')
    parser.add_argument('fea', help='feature file')
    args = parser.parse_args()

    with open(args.dict) as fdict:
        word_dict = load_dict(fdict)

    with open(args.fea) as ffea:
        word_fea = load_emb(ffea)

    while True:
        w1, w2 = raw_input("please input two words: ").split()
        w1_id = get_wordidx(w1, word_dict)
        w2_id = get_wordidx(w2, word_dict)
        if w1_id == -1 or w2_id == -1:
            continue
        print 'similarity: %s' % (calcos(w1_id, w2_id, word_fea))