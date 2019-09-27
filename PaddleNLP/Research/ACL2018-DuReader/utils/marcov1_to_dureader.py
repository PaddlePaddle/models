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

import sys
import json
import pandas as pd


def trans(input_js):
    output_js = {}
    output_js['question'] = input_js['query']
    output_js['question_type'] = input_js['query_type']
    output_js['question_id'] = input_js['query_id']
    output_js['fact_or_opinion'] = ""
    output_js['documents'] = []
    for para_id, para in enumerate(input_js['passages']):
        doc = {}
        doc['title'] = ""
        if 'is_selected' in para:
            doc['is_selected'] = True if para['is_selected'] != 0 else False
        doc['paragraphs'] = [para['passage_text']]
        output_js['documents'].append(doc)

    if 'answers' in input_js:
        output_js['answers'] = input_js['answers']
    return output_js


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: marcov1_to_dureader.py <input_path>')
        exit()

    df = pd.read_json(sys.argv[1])
    for row in df.iterrows():
        marco_js = json.loads(row[1].to_json())
        dureader_js = trans(marco_js)
        print(json.dumps(dureader_js))
