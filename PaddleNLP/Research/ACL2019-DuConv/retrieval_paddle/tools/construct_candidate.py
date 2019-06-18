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
File: construct_candidate.py
"""

from __future__ import print_function
import sys
import json
import random
import collections

reload(sys)
sys.setdefaultencoding('utf8')


def load_candidate_set(candidate_set_file):
    """
    load candidate set
    """
    candidate_set = []
    for line in open(candidate_set_file):
        candidate_set.append(json.loads(line.strip(), encoding="utf-8"))

    return candidate_set


def candidate_slection(candidate_set, knowledge_dict, slot_dict, candidate_num=10):
    """
    candidate slection
    """
    random.shuffle(candidate_set)
    candidate_legal = []
    for candidate in candidate_set:
        is_legal = True
        for slot in slot_dict:
            if slot in ["topic_a", "topic_b"]:
                continue
            if slot in candidate:
                if slot not in knowledge_dict:
                    is_legal = False
                    break
                w_ = random.choice(knowledge_dict[slot])
                candidate = candidate.replace(slot, w_)

        for slot in ["topic_a", "topic_b"]:
            if slot in candidate:
                if slot not in knowledge_dict:
                    is_legal = False
                    break
                w_ = random.choice(knowledge_dict[slot])
                candidate = candidate.replace(slot, w_)

        if is_legal and candidate not in candidate_legal:
            candidate_legal.append(candidate)

        if len(candidate_legal) >= candidate_num:
            break

    return candidate_legal


def get_candidate_for_conversation(conversation, candidate_set, candidate_num=10):
    """
    get candidate for conversation
    """
    candidate_set_gener, candidate_set_mater, candidate_set_list, slot_dict = candidate_set

    chat_path = conversation["goal"]
    knowledge = conversation["knowledge"]
    history = conversation["history"]

    topic_a = chat_path[0][1]
    topic_b = chat_path[0][2]
    domain_a = None
    domain_b = None
    knowledge_dict = {"topic_a":[topic_a], "topic_b":[topic_b]}
    for i, [s, p, o] in enumerate(knowledge):
        p_key = ""
        if topic_a.replace(' ', '') == s.replace(' ', ''):
            p_key = "topic_a_" + p.replace(' ', '')
            if u"领域" == p:
                domain_a = o
        elif topic_b.replace(' ', '') == s.replace(' ', ''):
            p_key = "topic_b_" + p.replace(' ', '')
            if u"领域" == p:
                domain_b = o

        if p_key == "":
            continue

        if p_key in knowledge_dict:
            knowledge_dict[p_key].append(o)
        else:
            knowledge_dict[p_key] = [o]

    assert domain_a is not None and domain_b is not None

    key = '_'.join([domain_a, domain_b, str(len(history))])

    candidate_legal = []
    if key in candidate_set_gener:
        candidate_legal.extend(candidate_slection(candidate_set_gener[key],
                                                  knowledge_dict, slot_dict,
                                                  candidate_num = candidate_num - len(candidate_legal)))

    if len(candidate_legal) < candidate_num and key in candidate_set_mater:
        candidate_legal.extend(candidate_slection(candidate_set_mater[key],
                                                  knowledge_dict, slot_dict,
                                                  candidate_num = candidate_num - len(candidate_legal)))

    if len(candidate_legal) < candidate_num:
        candidate_legal.extend(candidate_slection(candidate_set_list,
                                                  knowledge_dict, slot_dict,
                                                  candidate_num = candidate_num - len(candidate_legal)))

    return candidate_legal


def construct_candidate_for_corpus(corpus_file, candidate_set_file, candidate_file, candidate_num=10):
    """
    construct candidate for corpus

    case of data in corpus_file:
    {
        "goal": [["START", "休 · 劳瑞", "蕾切儿 · 哈伍德"]],
        "knowledge": [["休 · 劳瑞", "评论", "完美 的 男人"]],
        "history": ["你 对 明星 有没有 到 迷恋 的 程度 呢 ？",
                    "一般 吧 ， 毕竟 年纪 不 小 了 ， 只是 追星 而已 。"]
    }

    case of data in candidate_file:
    {
        "goal": [["START", "休 · 劳瑞", "蕾切儿 · 哈伍德"]],
        "knowledge": [["休 · 劳瑞", "评论", "完美 的 男人"]],
        "history": ["你 对 明星 有没有 到 迷恋 的 程度 呢 ？",
                    "一般 吧 ， 毕竟 年纪 不 小 了 ， 只是 追星 而已 。"],
        "candidate": ["我 说 的 是 休 · 劳瑞 。",
                      "我 说 的 是 休 · 劳瑞 。"]
    }
    """
    candidate_set = load_candidate_set(candidate_set_file)
    fout_text = open(candidate_file, 'w')
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            conversation = json.loads(line.strip(), encoding="utf-8", \
                                 object_pairs_hook=collections.OrderedDict)
            candidates = get_candidate_for_conversation(conversation,
                                                        candidate_set,
                                                        candidate_num=candidate_num)
            conversation["candidate"] = candidates

            conversation = json.dumps(conversation, ensure_ascii=False, encoding="utf-8")
            fout_text.write(conversation + "\n")

    fout_text.close()


def main():
    """
    main
    """
    construct_candidate_for_corpus(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
