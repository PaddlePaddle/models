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
File: build_candidate_set_from_corpus.py
"""

from __future__ import print_function
import sys
import json
import random
import collections

reload(sys)
sys.setdefaultencoding('utf8')


def build_candidate_set_from_corpus(corpus_file, candidate_set_file):
    """
    build candidate set from corpus
    """
    candidate_set_gener = {}
    candidate_set_mater = {}
    candidate_set_list = []
    slot_dict = {"topic_a": 1, "topic_b": 1}
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            conversation = json.loads(line.strip(), encoding="utf-8", \
                                      object_pairs_hook=collections.OrderedDict)

            chat_path = conversation["goal"]
            knowledge = conversation["knowledge"]
            session = conversation["conversation"]

            topic_a = chat_path[0][1]
            topic_b = chat_path[0][2]
            domain_a = None
            domain_b = None
            cover_att_list = [[["topic_a", topic_a], ["topic_b", topic_b]]] * len(session)
            for j, [s, p, o] in enumerate(knowledge):
                p_key = ""
                if topic_a.replace(' ', '') == s.replace(' ', ''):
                    p_key = "topic_a_" + p.replace(' ', '')
                    if u"领域" == p:
                        domain_a = o
                elif topic_b.replace(' ', '') == s.replace(' ', ''):
                    p_key = "topic_b_" + p.replace(' ', '')
                    if u"领域" == p:
                        domain_b = o

                for k, utterance in enumerate(session):
                    if k % 2 == 1: continue
                    if o in utterance and o != topic_a and o != topic_b and p_key != "":
                        cover_att_list[k].append([p_key, o])

                slot_dict[p_key] = 1

            assert domain_a is not None and domain_b is not None

            for j, utterance in enumerate(session):
                if j % 2 == 1: continue
                key = '_'.join([domain_a, domain_b, str(j)])

                cover_att = sorted(cover_att_list[j], lambda x, y: cmp(len(x[1]), len(y[1])), reverse=True)

                utterance_gener = utterance
                for [p_key, o] in cover_att:
                    utterance_gener = utterance_gener.replace(o, p_key)

                if "topic_a_topic_a_" not in utterance_gener and \
                   "topic_a_topic_b_" not in utterance_gener and \
                   "topic_b_topic_a_" not in utterance_gener and \
                   "topic_b_topic_b_" not in utterance_gener:
                    if key in candidate_set_gener:
                        candidate_set_gener[key].append(utterance_gener)
                    else:
                        candidate_set_gener[key] = [utterance_gener]
                
                utterance_mater = utterance
                for [p_key, o] in [["topic_a", topic_a], ["topic_b", topic_b]]:
                    utterance_mater = utterance_mater.replace(o, p_key)

                if key in candidate_set_mater:
                    candidate_set_mater[key].append(utterance_mater)
                else:
                    candidate_set_mater[key] = [utterance_mater]

                candidate_set_list.append(utterance_mater)

    fout = open(candidate_set_file, 'w')
    fout.write(json.dumps(candidate_set_gener, ensure_ascii=False, encoding="utf-8") + "\n")
    fout.write(json.dumps(candidate_set_mater, ensure_ascii=False, encoding="utf-8") + "\n")
    fout.write(json.dumps(candidate_set_list, ensure_ascii=False, encoding="utf-8") + "\n")
    fout.write(json.dumps(slot_dict, ensure_ascii=False, encoding="utf-8"))
    fout.close()


def main():
    """
    main
    """
    build_candidate_set_from_corpus(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")