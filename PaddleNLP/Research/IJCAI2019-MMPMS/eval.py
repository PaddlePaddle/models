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

import codecs
import sys
import json

from random import shuffle
from mmpms.utils.metrics import Metric, bleu, distinct

NUM_MULTI_RESPONSES = 5


def evaluate_generation(results):

    tgt = [result["response"].split(" ") for result in results]
    tgt_multi = [x for x in tgt for _ in range(NUM_MULTI_RESPONSES)]
    preds = [
        list(map(lambda s: s.split(" "), result["preds"])) for result in results
    ]

    # Shuffle predictions
    for n in range(len(preds)):
        shuffle(preds[n])

    # Single response generation
    pred = [ps[0] for ps in preds]
    bleu1, bleu2 = bleu(pred, tgt)
    dist1, dist2 = distinct(pred)
    print("Random 1 candidate:   " + "BLEU-1/2: {:.3f}/{:.3f}   ".format(
        bleu1, bleu2) + "DIST-1/2: {:.3f}/{:.3f}".format(dist1, dist2))

    # Multiple response generation
    pred = [ps[:5] for ps in preds]
    pred = [p for ps in pred for p in ps]
    bleu1, bleu2 = bleu(pred, tgt_multi)
    dist1, dist2 = distinct(pred)
    print("Random {} candidates:   ".format(
        NUM_MULTI_RESPONSES) + "BLEU-1/2: {:.3f}/{:.3f}   ".format(bleu1, bleu2)
          + "DIST-1/2: {:.3f}/{:.3f}".format(dist1, dist2))


def main():
    result_file = sys.argv[1]
    with codecs.open(result_file, "r", encoding="utf-8") as fp:
        results = json.load(fp)
    evaluate_generation(results)


if __name__ == '__main__':
    main()
