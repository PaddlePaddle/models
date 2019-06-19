#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module convert MRQA official data to SQuAD format
"""

import json
import argparse
import re


def reader(filename):
    """
    This function read a MRQA data file.
    :param filename: name of a MRQA data file.
    :return: original samples of a MRQA data file.
    """
    with open(filename) as fin:
        for lidx, line in enumerate(fin):
            if lidx == 0:
                continue
            sample = json.loads(line.strip())
            yield sample


def to_squad_para_train(sample):
    """
    This function convert training data from MRQA format to SQuAD format.
    :param sample: one sample in MRQA format.
    :return: paragraphs in SQuAD format.
    """
    squad_para = dict()
    context = sample['context']
    context = re.sub(r'\[TLE\]|\[DOC\]|\[PAR\]', '[SEP]', context)
    # replace special tokens to [SEP] to avoid UNK in BERT
    squad_para['context'] = context
    qas = []
    for qa in sample['qas']:
        text = qa['detected_answers'][0]['text']
        new_start = context.find(text)
        # Try to find an exact match (without normalization) of the reference answer.
        # Some articles like {a|an|the} my get lost in the original spans.
        # E.g. the reference answer is "The London Eye",
        # while the original span may only contain "London Eye" due to normalization.
        new_end = new_start + len(text) - 1
        org_start = qa['detected_answers'][0]['char_spans'][0][0]
        org_end = qa['detected_answers'][0]['char_spans'][0][1]
        if new_start == -1 or len(text) < 8:
            # If no exact match (without normalization) can be found or reference answer is too short
            # (e.g. only contain a character "c", which will cause problems using find),
            # use the original span in MRQA dataset.
            answer = {
                'text': squad_para['context'][org_start:org_end + 1],
                'answer_start': org_start
            }
            answer_start = org_start
            answer_end = org_end
        else:
            answer = {
                'text': text,
                'answer_start': new_start
            }
            answer_start = new_start
            answer_end = new_end
        # A sanity check
        try:
            assert answer['text'].lower() == squad_para['context'][answer_start:answer_end + 1].lower()
        except AssertionError:
            print(answer['text'])
            print(squad_para['context'][answer_start:answer_end + 1])
            continue
        squad_qa = {
            'question': qa['question'],
            'id': qa['qid'],
            'answers': [answer]
        }
        qas.append(squad_qa)
    squad_para['qas'] = qas
    return squad_para


def to_squad_para_dev(sample):
    """
    This function convert development data from MRQA format to SQuAD format.
    :param sample: one sample in MRQA format.
    :return: paragraphs in SQuAD format.
    """

    squad_para = dict()
    context = sample['context']
    context = re.sub(r'\[TLE\]|\[DOC\]|\[PAR\]', '[SEP]', context)
    squad_para['context'] = context
    qas = []
    for qa in sample['qas']:
        org_answers = qa['answers']
        answers = []
        for org_answer in org_answers:
            answer = {
                'text': org_answer,
                'answer_start': -1
            }
            answers.append(answer)
        squad_qa = {
            'question': qa['question'],
            'id': qa['qid'],
            'answers': answers
        }
        qas.append(squad_qa)
    squad_para['qas'] = qas
    return squad_para


def doc_wrapper(squad_para, title=""):
    """
    This function wrap paragraphs into a document.
    :param squad_para: paragraphs in SQuAD format.
    :param title: the title of paragraphs.
    :return: wrap of title and paragraphs
    """
    squad_doc = {
        'title': title,
        'paragraphs': [squad_para]
    }
    return squad_doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='the input file')
    parser.add_argument('--dev', action='store_true', help='convert devset')
    args = parser.parse_args()
    file_prefix = args.input[0:-6]
    squad = {
        'data': [],
        'version': "1.1"
    }
    to_squad_para = to_squad_para_dev if args.dev else to_squad_para_train
    for org_sample in reader(args.input):
        para = to_squad_para(org_sample)
        doc = doc_wrapper(para)
        squad['data'].append(doc)
    with open('{}.raw.json'.format(file_prefix), 'w') as fout:
        json.dump(squad, fout, indent=4)
