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
#coding=utf8

import os, sys, json
import nltk


def _nltk_tokenize(sequence):
    tokens = nltk.word_tokenize(sequence)

    cur_char_offset = 0
    token_offsets = []
    token_words = []
    for token in tokens:
        cur_char_offset = sequence.find(token, cur_char_offset)
        token_offsets.append(
            [cur_char_offset, cur_char_offset + len(token) - 1])
        token_words.append(token)
    return token_offsets, token_words


def segment(input_js):
    _, input_js['segmented_question'] = _nltk_tokenize(input_js['question'])
    for doc_id, doc in enumerate(input_js['documents']):
        doc['segmented_title'] = []
        doc['segmented_paragraphs'] = []
        for para_id, para in enumerate(doc['paragraphs']):
            _, seg_para = _nltk_tokenize(para)
            doc['segmented_paragraphs'].append(seg_para)
    if 'answers' in input_js:
        input_js['segmented_answers'] = []
        for answer_id, answer in enumerate(input_js['answers']):
            _, seg_answer = _nltk_tokenize(answer)
            input_js['segmented_answers'].append(seg_answer)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: tokenize_data.py <input_path>')
        exit()

    nltk.download('punkt')

    for line in open(sys.argv[1]):
        dureader_js = json.loads(line.strip())
        segment(dureader_js)
        print(json.dumps(dureader_js))
