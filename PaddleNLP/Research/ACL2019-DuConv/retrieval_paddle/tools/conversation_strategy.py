#!/usr/bin/env python
# -*- coding: utf-8 -*-
######################################################################
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
######################################################################
"""
File: conversation_strategy.py
"""

from __future__ import print_function

import sys
sys.path.append("../")
import interact
from tools.convert_conversation_corpus_to_model_text import preprocessing_for_one_conversation
from tools.construct_candidate import load_candidate_set

reload(sys)
sys.setdefaultencoding('utf8')


def load():
    """
    load
    """
    return interact.load_model(), load_candidate_set("../data/candidate_set.txt")


def predict(model, text):
    """
    predict
    """
    model, candidate_set = model
    model_text, candidates = \
        preprocessing_for_one_conversation(text.strip(),
                                           candidate_set=candidate_set,
                                           candidate_num=50,
                                           use_knowledge=True,
                                           topic_generalization=True,
                                           for_predict=True)

    for i, text_ in enumerate(model_text):
        score = interact.predict(model, text_, task_name="match_kn_gene")
        candidates[i] = [candidates[i], score]

    candidate_legal = sorted(candidates, key=lambda item: item[1], reverse=True)
    return candidate_legal[0][0]


def main():
    """
    main
    """
    model = load()
    for line in sys.stdin:
        response = predict(model, line.strip())
        print(response)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
