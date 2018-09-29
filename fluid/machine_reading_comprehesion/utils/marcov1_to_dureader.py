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
