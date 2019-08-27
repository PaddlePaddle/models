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

import pickle
import argparse
import os
import nltk
import logging
import string
from tqdm import tqdm
from nltk.corpus import wordnet as wn

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_token', type=str, default='../tokenization_record/tokens/train.tokenization.uncased.data', help='token file of train set')
    parser.add_argument('--eval_token', type=str, default='../tokenization_record/tokens/dev.tokenization.uncased.data', help='token file of dev set')
    parser.add_argument('--output_dir', type=str, default='output_record/', help='output directory')
    parser.add_argument('--no_stopwords', action='store_true', help='ignore stopwords')
    parser.add_argument('--ignore_length', type=int, default=0, help='ignore words with length <= ignore_length')
    args = parser.parse_args()

    # initialize mapping from offset id to wn18 synset name
    offset_to_wn18name_dict = {} 
    fin = open('wordnet-mlj12-definitions.txt')
    for line in fin:
        info = line.strip().split('\t')
        offset_str, synset_name = info[0], info[1]
        offset_to_wn18name_dict[offset_str] = synset_name
    logger.info('Finished loading wn18 definition file.')
        

    # load pickled samples
    logger.info('Begin to load tokenization results...')
    train_samples = pickle.load(open(args.train_token, 'rb'))
    dev_samples = pickle.load(open(args.eval_token, 'rb'))
    logger.info('Finished loading tokenization results.')
    
    # build token set
    all_token_set = set()
    for sample in train_samples + dev_samples:
        for token in sample['query_tokens'] + sample['document_tokens']:
            all_token_set.add(token)
    logger.info('Finished making tokenization results into token set.')

    # load stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    logger.info('Finished loading stopwords list.')

    # retrive synsets
    logger.info('Begin to retrieve synsets...')
    token2synset = dict()
    stopword_cnt = 0
    punctuation_cnt = 0
    for token in tqdm(all_token_set):
        if token in set(string.punctuation):
            logger.info('{} is punctuation, skipped!'.format(token))
            punctuation_cnt += 1
            continue        
        if args.no_stopwords and token in stopwords:
            logger.info('{} is stopword, skipped!'.format(token))
            stopword_cnt += 1
            continue
        if args.ignore_length > 0 and len(token) <= args.ignore_length:
            logger.info('{} is too short, skipped!'.format(token))
            continue
        synsets = wn.synsets(token)
        wn18synset_names = []
        for synset in synsets:
            offset_str = str(synset.offset()).zfill(8)
            if offset_str in offset_to_wn18name_dict:
                wn18synset_names.append(offset_to_wn18name_dict[offset_str])
        if len(wn18synset_names) > 0:
            token2synset[token] = wn18synset_names
    logger.info('Finished retrieving sysnets.')
    logger.info('{} / {} tokens retrieved at lease 1 synset. {} stopwords and {} punctuations skipped.'.format(len(token2synset), len(all_token_set), stopword_cnt, punctuation_cnt))
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'retrived_synsets.data'), 'wb') as fout:
        pickle.dump(token2synset, fout)    
    logger.info('Finished dumping retrieved synsets.')

if __name__ == '__main__':
    main()
