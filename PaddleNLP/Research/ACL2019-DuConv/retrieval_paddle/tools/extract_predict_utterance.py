#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: extract_predict_utterance.py
"""

from __future__ import print_function

import sys
import json
import collections

reload(sys)
sys.setdefaultencoding('utf8')


def extract_predict_utterance(sample_file, score_file, output_file):
    """
    convert_result_for_eval
    """
    sample_list = [line.strip() for line in open(sample_file, 'r')]
    score_list = [line.strip() for line in open(score_file, 'r')]

    fout = open(output_file, 'w')
    index = 0
    for i, sample in enumerate(sample_list):
        sample = json.loads(sample, encoding="utf-8", \
                              object_pairs_hook=collections.OrderedDict)

        candidates = sample["candidate"]
        scores = score_list[index: index + len(candidates)]

        pridict = candidates[0]
        max_score = float(scores[0])
        for j, score in enumerate(scores):
            score = float(score)
            if score > max_score:
                pridict = candidates[j]
                max_score = score

        if "response" in sample:
            response = sample["response"]
            fout.write(pridict + "\t" + response + "\n")
        else:
            fout.write(pridict + "\n")

        index = index + len(candidates)

    fout.close()


def main():
    """
    main
    """
    extract_predict_utterance(sys.argv[1],
                              sys.argv[2],
                              sys.argv[3])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
