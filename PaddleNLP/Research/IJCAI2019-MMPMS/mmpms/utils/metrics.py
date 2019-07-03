#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
################################################################################

from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction


class Metric(object):
    def __init__(self):
        self.reset()

    def update(self, val, num=1):
        self.val = float(val)
        self.num += num
        p = num / self.num
        self.avg = self.val * p + self.avg * (1 - p)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.num = 0

    def __repr__(self):
        return "Metric(val={}, avg={}, num={})".format(self.val, self.avg,
                                                       self.num)

    def state_dict(self):
        return {"val": self.val, "avg": self.avg, "num": self.num}


def distinct(seqs):
    batch_size = len(seqs)
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    dist_1 = (len(unigrams_all) + 1e-12) / (sum(unigrams_all.values()) + 1e-5)
    dist_2 = (len(bigrams_all) + 1e-12) / (sum(bigrams_all.values()) + 1e-5)
    return dist_1, dist_2


def bleu(hyps, refs):
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref],
                hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref],
                hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_2.append(score)
    bleu_1 = sum(bleu_1) / len(bleu_1)
    bleu_2 = sum(bleu_2) / len(bleu_2)
    return bleu_1, bleu_2
