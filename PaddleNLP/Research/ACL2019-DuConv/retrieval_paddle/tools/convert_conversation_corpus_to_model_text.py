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
File: convert_conversation_corpus_to_model_text.py
"""

from __future__ import print_function
import sys
sys.path.append("./")
import re
import json
import collections
from tools.construct_candidate import get_candidate_for_conversation

reload(sys)
sys.setdefaultencoding('utf8')

def parser_char_for_word(word):
    """
    parser char for word
    """
    if word.isdigit():
        word = word.decode('utf8')
    for i in range(len(word)):
        if word[i] >= u'\u4e00' and word[i] <= u'\u9fa5':
            word_out = " ".join([t.encode('utf8') for t in word])
            word_out = re.sub(" +", " ", word_out)
            return word_out
    return word.encode('utf8')


def parser_char_for_text(text):
    """
    parser char for text
    """
    words = text.strip().split()
    for i, word in enumerate(words):
        words[i] = parser_char_for_word(word)
    return re.sub(" +", " ", ' '.join(words))


def topic_generalization_for_text(text, topic_list):
    """
    topic generalization for text
    """
    for key, value in topic_list:
        text = text.replace(value, key)

    return text


def topic_generalization_for_list(text_list, topic_list):
    """
    topic generalization for list
    """
    for i, text in enumerate(text_list):
        text_list[i] = topic_generalization_for_text(text, topic_list)

    return text_list


def preprocessing_for_one_conversation(text, \
                                       candidate_set=None, \
                                       candidate_num=10, \
                                       use_knowledge=True, \
                                       topic_generalization=False, \
                                       for_predict=True):
    """
    preprocessing for one conversation
    """

    conversation = json.loads(text.strip(), encoding="utf-8", \
                              object_pairs_hook=collections.OrderedDict)

    goal = conversation["goal"]
    knowledge = conversation["knowledge"]
    history = conversation["history"]
    if not for_predict:
        response = conversation["response"]


    topic_a = goal[0][1]
    topic_b = goal[0][2]
    for i, [s, p, o] in enumerate(knowledge):
        if u"领域" == p:
            if topic_a == s:
                domain_a = o
            elif topic_b == s:
                domain_b = o

    topic_dict = {}
    if u"电影" == domain_a:
        topic_dict["video_topic_a"] = topic_a
    else:
        topic_dict["person_topic_a"] = topic_a

    if u"电影" == domain_b:
        topic_dict["video_topic_b"] = topic_b
    else:
        topic_dict["person_topic_b"] = topic_b

    if "candidate" in conversation:
        candidates = conversation["candidate"]
    else:
        assert candidate_num > 0 and candidate_set is not None
        candidates = get_candidate_for_conversation(conversation,
                                                    candidate_set,
                                                    candidate_num=candidate_num)

    if topic_generalization:
        topic_list = sorted(topic_dict.items(), key=lambda item: len(item[1]), reverse=True)

        goal = [topic_generalization_for_list(spo, topic_list) for spo in goal]

        knowledge = [topic_generalization_for_list(spo, topic_list) for spo in knowledge]

        history = [topic_generalization_for_text(utterance, topic_list)
                                    for utterance in history]

        for i, candidate in enumerate(candidates):
            candidates[i] = topic_generalization_for_text(candidate, topic_list)

        if not for_predict:
            response = topic_generalization_for_text(response, topic_list)

    goal = ' [PATH_SEP] '.join([parser_char_for_text(' '.join(spo))
                                for spo in goal])
    knowledge = ' [KN_SEP] '.join([parser_char_for_text(' '.join(spo))
                                   for spo in knowledge])
    history = ' [INNER_SEP] '.join([parser_char_for_text(utterance)
                                    for utterance in history]) \
                                        if len(history) > 0 else '[START]'

    model_text = []

    for candidate in candidates:
        candidate = parser_char_for_text(candidate)
        if use_knowledge:
            text_ = '\t'.join(["0", history, candidate, goal, knowledge])
        else:
            text_ = '\t'.join(["0", history, candidate])

        text_ = re.sub(" +", " ", text_)
        model_text.append(text_)

    if not for_predict:
        candidates.append(response)
        response = parser_char_for_text(response)
        if use_knowledge:
            text_ = '\t'.join(["1", history, response, goal, knowledge])
        else:
            text_ = '\t'.join(["1", history, response])

        text_ = re.sub(" +", " ", text_)
        model_text.append(text_)

    return model_text, candidates


def convert_conversation_corpus_to_model_text(corpus_file, 
                                              text_file,
                                              use_knowledge=True,
                                              topic_generalization=False,
                                              for_predict=True):
    """
    convert conversation corpus to model text
    """
    fout_text = open(text_file, 'w')
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            model_text, _ = preprocessing_for_one_conversation(
                line.strip(), 
                candidate_set=None,
                candidate_num=0,
                use_knowledge=use_knowledge,
                topic_generalization=topic_generalization,
                for_predict=for_predict)

            for text in model_text:
                fout_text.write(text + "\n")

    fout_text.close()


def main():
    """
    main
    """
    convert_conversation_corpus_to_model_text(sys.argv[1],
                                              sys.argv[2],
                                              int(sys.argv[3]) > 0,
                                              int(sys.argv[4]) > 0,
                                              int(sys.argv[5]) > 0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
