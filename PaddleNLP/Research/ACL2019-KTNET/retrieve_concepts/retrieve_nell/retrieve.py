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

# This script retrieve related NELL entities and their concepts for each named-entity in ReCoRD
# 1. transform ReCoRD entity from word sequences into strings (use _ to replace whitespace and eliminate punc)
# 2. preprocess NELL entity name (remove front 'n' for NELL entities when digit is in the beginning and additional _)
# 3. for ReCoRD entities with more than one token, use exact match
# 4. for one-word ReCoRD entities, do wordnet lemmatization before matching (and matching by both raw and morphed forms)
# 5. in a passage, if entity A is a suffix of entity B, use B's categories instead

import pickle
import logging
import string
import argparse
import os
import nltk
from collections import namedtuple
from tqdm import tqdm
from nltk.corpus import wordnet as wn

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# remove category part of NELL entities, digit prefix 'n' and additional '_'
def preprocess_nell_ent_name(raw_name):
    ent_name = raw_name.split(':')[-1]
    digits = set(string.digits)
    if ent_name.startswith('n') and all([char in digits for char in ent_name.split('_')[0][1:]]):
        ent_name = ent_name[1:]
    ent_name = "_".join(filter(lambda x:len(x) > 0, ent_name.split('_')))
    return ent_name

puncs = set(string.punctuation)
def preprocess_record_ent_name(raw_token_seq):
    return "_".join(filter(lambda x:x not in puncs, raw_token_seq))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_token', type=str, default='../tokenization_record/tokens/train.tokenization.uncased.data', help='token file of train set')
    parser.add_argument('--eval_token', type=str, default='../tokenization_record/tokens/dev.tokenization.uncased.data', help='token file of dev set')
    parser.add_argument('--score_threshold', type=float, default=0.9, help='only keep generalizations relations with score >= threshold')    
    parser.add_argument('--output_dir', type=str, default='output_record/', help='output directory')
    args = parser.parse_args()

    # make output directory if not exist
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # load set of concepts with pre-trained embedding
    concept_set = set()
    with open('nell_concept_list.txt') as fin:
        for line in fin:
            concept_name = line.strip()
            concept_set.add(concept_name)

    # read nell csv file and build NELL entity to category dict
    logger.info('Begin to read NELL csv...')
    fin = open('NELL.08m.1115.esv.csv')
    nell_ent_to_cpt = {}
    nell_ent_to_fullname = {}

    header = True
    for line in fin:
        if header:
            header = False
            continue
        line = line.strip()
        items = line.split('\t')
        if items[1] == 'generalizations' and float(items[4]) >= args.score_threshold:
            nell_ent_name = preprocess_nell_ent_name(items[0])
            category = items[2]
            if nell_ent_name not in nell_ent_to_cpt:
                nell_ent_to_cpt[nell_ent_name] = set()
                nell_ent_to_fullname[nell_ent_name] = set()
            nell_ent_to_cpt[nell_ent_name].add(category)
            nell_ent_to_fullname[nell_ent_name].add(items[0])
    logger.info('Finished reading NELL csv.')

    # load record dataset
    logger.info('Begin to load tokenization results...')
    train_samples = pickle.load(open(args.train_token, 'rb'))
    dev_samples = pickle.load(open(args.eval_token, 'rb'))
    logger.info('Finished loading tokenization results.')

    # build record entity set
    record_ent_set = set()
    for sample in train_samples + dev_samples:
        query_tokens = sample['query_tokens']
        document_tokens = sample['document_tokens']
        for entity_info in sample['document_entities']:
            entity_token_seq = document_tokens[entity_info[1]: entity_info[2] + 1]
            record_ent_set.add(preprocess_record_ent_name(entity_token_seq))
        for entity_info in sample['query_entities']:
            entity_token_seq = query_tokens[entity_info[1]: entity_info[2] + 1]
            record_ent_set.add(preprocess_record_ent_name(entity_token_seq))   
    logger.info('Finished making tokenization results into entity set.')        

    # do mapping
    record_ent_to_cpt = {}
    record_ent_to_nell_ent = {}
    for record_ent in tqdm(record_ent_set):
        cpt, nell_ent = set(), set()
        if record_ent in nell_ent_to_cpt:
            cpt.update(nell_ent_to_cpt[record_ent])
            nell_ent.update(nell_ent_to_fullname[record_ent])
        # length is 1, do morphy
        if '_' not in record_ent:
            for pos_tag in ['n', 'v', 'a', 'r']:
                morph = wn.morphy(record_ent, pos_tag)
                if morph is not None and morph in nell_ent_to_cpt:
                    cpt.update(nell_ent_to_cpt[morph])
                    nell_ent.update(nell_ent_to_fullname[morph])
        record_ent_to_cpt[record_ent] = cpt
        record_ent_to_nell_ent[record_ent] = nell_ent
    logger.info('Finished matching record entities to nell entities.')
    
    # map the record entity in the set back to passage
    logger.info('Begin to generate output file...')
    _TempRectuple = namedtuple('entity_record', [
                               'entity_string', 'start', 'end', 'retrieved_concepts', 'retrieved_entities'])
    for outfn, samples in zip(('{}.retrieved_nell_concepts.data'.format(prefix) for prefix in ('train', 'dev')), (train_samples, dev_samples)):
        all_outputs = []
        for sample in tqdm(samples):
            doc_entities = []
            document_tokens = sample['document_tokens']
            for entity_info in sample['document_entities']:
                entity_token_seq = document_tokens[entity_info[1]: entity_info[2] + 1]
                entity_whitespace_str = " ".join(entity_token_seq)
                entity_retrieve_str = preprocess_record_ent_name(
                    entity_token_seq)
                doc_entities.append(_TempRectuple(
                    entity_whitespace_str, entity_info[1], entity_info[2], record_ent_to_cpt[entity_retrieve_str], record_ent_to_nell_ent[entity_retrieve_str]))
            query_entities = []
            query_tokens = sample['query_tokens']
            for entity_info in sample['query_entities']:
                entity_token_seq = query_tokens[entity_info[1]: entity_info[2] + 1]
                entity_whitespace_str = " ".join(entity_token_seq)
                entity_retrieve_str = preprocess_record_ent_name(
                    entity_token_seq)
                query_entities.append(_TempRectuple(
                    entity_whitespace_str, entity_info[1], entity_info[2], record_ent_to_cpt[entity_retrieve_str], record_ent_to_nell_ent[entity_retrieve_str]))                
            
            # perform suffix replacement rule (eg. use the result of "Donald Trump" to replace "Trump" in the passage)
            doc_entities_final, query_entities_final = [], []
            for entities, entities_final in zip((doc_entities, query_entities), (doc_entities_final, query_entities_final)):
                for trt in entities:
                    new_nell_cpt_set, new_nell_ent_set = set(), set()
                    for other_trt in doc_entities + query_entities:
                        if other_trt.entity_string != trt.entity_string and other_trt.entity_string.endswith(trt.entity_string):
                            new_nell_cpt_set.update(other_trt.retrieved_concepts)
                            new_nell_ent_set.update(other_trt.retrieved_entities)
                    # no need to replace
                    if len(new_nell_cpt_set) == 0:
                        new_nell_cpt_set = trt.retrieved_concepts
                        new_nell_ent_set = trt.retrieved_entities
                    new_nell_cpt_set = new_nell_cpt_set & concept_set # filter concepts with pretrained embedding
                    entities_final.append({
                        'entity_string': trt.entity_string,
                        'token_start': trt.start,
                        'token_end': trt.end,
                        'retrieved_concepts': list(new_nell_cpt_set),
                        'retrieved_entities': list(new_nell_ent_set),
                    })
            
            all_outputs.append({
                'id': sample['id'],
                'document_entities': doc_entities_final,
                'query_entities': query_entities_final,
            })
        pickle.dump(all_outputs, open(os.path.join(args.output_dir, outfn), 'wb'))
    logger.info('Output retrieved results have been dumped.')
        
if __name__ == '__main__':
    main()
