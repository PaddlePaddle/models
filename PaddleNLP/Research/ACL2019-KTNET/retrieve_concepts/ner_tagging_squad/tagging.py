# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright 2019 Baidu.com, Inc. All Rights Reserved
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

# This script perform NER tagging for raw SQuAD datasets
# All the named entites found in question and context are recorded with their offsets in the output file
# CoreNLP is used for NER tagging

import os
import json
import argparse
import logging
import urllib
import sys
from tqdm import tqdm
from pycorenlp import StanfordCoreNLP

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default='output', type=str, 
                        help="The output directory to store tagging results.")
    parser.add_argument("--train_file", default='../../data/SQuAD/train-v1.1.json', type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default='../../data/SQuAD/dev-v1.1.json', type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    return parser.parse_args()

# transform corenlp tagging output into entity list
# some questions begins with whitespaces and they are striped by corenlp, thus begin offset should be added.
def parse_output(text, tagging_output, begin_offset=0): 
    entities = []
    select_states = ['ORGANIZATION', 'PERSON', 'MISC', 'LOCATION']
    for sent in tagging_output['sentences']:
        state = 'O'
        start_pos, end_pos = -1, -1
        for token in sent['tokens']:
            tag = token['ner']  
            if tag == 'O' and state != 'O':
                if state in select_states:
                    entities.append({'text': text[begin_offset + start_pos: begin_offset + end_pos], 'start': begin_offset + start_pos, 'end': begin_offset + end_pos - 1})
                state = 'O'
            elif tag != 'O':
                if state == tag:
                    end_pos = token['characterOffsetEnd']
                else:
                    if state in select_states:
                        entities.append({'text': text[begin_offset + start_pos: begin_offset + end_pos], 'start': begin_offset + start_pos, 'end': begin_offset + end_pos - 1})
                    state = tag
                    start_pos = token['characterOffsetBegin']
                    end_pos = token['characterOffsetEnd']
        if state in select_states:
            entities.append({'text': text[begin_offset + start_pos: begin_offset + end_pos], 'start': begin_offset + start_pos, 'end': begin_offset + end_pos - 1})
    return entities
                
def tagging(dataset, nlp):
    skip_context_cnt, skip_question_cnt = 0, 0
    for article in tqdm(dataset['data']):
        for paragraph in tqdm(article['paragraphs']):
            context = paragraph['context']
            context_tagging_output = nlp.annotate(urllib.parse.quote(context), properties={'annotators': 'ner', 'outputFormat': 'json'})
            # assert the context length is not changed
            if len(context.strip()) == context_tagging_output['sentences'][-1]['tokens'][-1]['characterOffsetEnd']:
                context_entities = parse_output(context, context_tagging_output, len(context) - len(context.lstrip()))
            else:
                context_entities = []
                skip_context_cnt += 1
                logger.info('Skipped context due to offset mismatch:')
                logger.info(context)
            paragraph['context_entities'] = context_entities
            for qa in tqdm(paragraph['qas']):
                question = qa['question']
                question_tagging_output = nlp.annotate(urllib.parse.quote(question), properties={'annotators': 'ner', 'outputFormat': 'json'})
                if len(question.strip()) == question_tagging_output['sentences'][-1]['tokens'][-1]['characterOffsetEnd']:
                    question_entities = parse_output(question, question_tagging_output, len(context) - len(context.lstrip()))
                else:
                    question_entities = []
                    skip_question_cnt += 1
                    logger.info('Skipped question due to offset mismatch:')
                    logger.info(question)                    
                qa['question_entities'] = question_entities
    logger.info('In total, {} contexts and {} questions are skipped...'.format(skip_context_cnt, skip_question_cnt))

if __name__ == '__main__':
    args = parse_args()
    
    # make output directory if not exist
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    # register corenlp server
    nlp = StanfordCoreNLP('http://localhost:9753')

    # load train and dev datasets
    ftrain = open(args.train_file, 'r', encoding='utf-8')
    trainset = json.load(ftrain)
    fdev = open(args.predict_file, 'r', encoding='utf-8')
    devset = json.load(fdev)
    
    for dataset, path, name in zip((trainset, devset), (args.train_file, args.predict_file), ('train', 'dev')):
        tagging(dataset, nlp)
        output_path = os.path.join(args.output_dir, "{}.tagged.json".format(os.path.basename(path)[:-5]))
        json.dump(dataset, open(output_path, 'w', encoding='utf-8'))
        logger.info('Finished tagging {} set'.format(name))
