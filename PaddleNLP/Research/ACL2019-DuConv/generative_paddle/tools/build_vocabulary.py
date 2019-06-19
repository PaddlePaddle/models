#!/usr/bin/env python
# -*- coding: utf-8 -*- 
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: build_vocabulary.py
"""

from __future__ import print_function

import sys
import re
from collections import Counter

reload(sys)
sys.setdefaultencoding('utf8')


def tokenize(s):
    """
    tokenize
    """
    s = re.sub('\d+', '<num>', s).lower()
    tokens = s.split(' ')
    return tokens


def build_vocabulary(corpus_file, vocab_file,
                     vocab_size=30004, min_frequency=0,
                     min_len=1, max_len=500):
    """
    build words dict
    """
    specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
    counter = Counter()
    for line in open(corpus_file, 'r'):
        src, tgt, knowledge = line.rstrip('\n').split('\t')[:3]
        filter_knowledge = []
        for sent in knowledge.split('\1'):
            filter_knowledge.append(' '.join(sent.split()[:max_len]))
        knowledge = ' '.join(filter_knowledge)

        src = tokenize(src)
        tgt = tokenize(tgt)
        knowledge = tokenize(knowledge)

        if len(src) < min_len or len(src) > max_len or \
           len(tgt) < min_len or len(tgt) > max_len:
            continue

        counter.update(src + tgt + knowledge)

    for tok in specials:
        del counter[tok]

    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    words_and_frequencies = [[tok, sys.maxint] for tok in specials] + words_and_frequencies
    words_and_frequencies = words_and_frequencies[:vocab_size]

    fout = open(vocab_file, 'w')
    for word, frequency in words_and_frequencies:
        if frequency < min_frequency:
            break
        fout.write(word + '\n')

    fout.close()


def main():
    """
    main
    """
    if len(sys.argv) < 3:
        print("Usage: " + sys.argv[0] + " corpus_file vocab_file")
        exit()

    build_vocabulary(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
