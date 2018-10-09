# -*- coding:utf8 -*-
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
Utility function to generate vocabulary file.
"""

import argparse
import sys
import json

from itertools import chain


def get_vocab(files, vocab_file):
    """
    Builds vocabulary file from field 'segmented_paragraphs'
    and 'segmented_question'.

    Args:
        files: A list of file names.
        vocab_file: The file that stores the vocabulary.
    """
    vocab = {}
    for f in files:
        with open(f, 'r') as fin:
            for line in fin:
                obj = json.loads(line.strip())
                paras = [
                    chain(*d['segmented_paragraphs']) for d in obj['documents']
                ]
                doc_tokens = chain(*paras)
                question_tokens = obj['segmented_question']
                for t in list(doc_tokens) + question_tokens:
                    vocab[t] = vocab.get(t, 0) + 1
    # output
    sorted_vocab = sorted(
        [(v, c) for v, c in vocab.items()], key=lambda x: x[1], reverse=True)
    with open(vocab_file, 'w') as outf:
        for w, c in sorted_vocab:
            print >> outf, '{}\t{}'.format(w.encode('utf8'), c)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--files',
        nargs='+',
        required=True,
        help='file list to count vocab from.')
    parser.add_argument(
        '--vocab', required=True, help='file to store counted vocab.')
    args = parser.parse_args()
    get_vocab(args.files, args.vocab)
