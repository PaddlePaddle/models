#!/usr/bin/env python
# -*- coding: utf-8 -*- 
######################################################################
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
######################################################################
"""
File: build_dict.py
"""

from __future__ import print_function
import sys

reload(sys)
sys.setdefaultencoding('utf8')


def build_dict(corpus_file, dict_file):
    """
    build words dict
    """
    dict = {}
    max_frequency = 1
    for line in open(corpus_file, 'r'):
        conversation = line.strip().split('\t')
        for i in range(1, len(conversation), 1):
            words = conversation[i].split(' ')
            for word in words:
                if word in dict:
                    dict[word] = dict[word] + 1
                    if dict[word] > max_frequency:
                        max_frequency = dict[word]
                else:
                    dict[word] = 1

    dict["[PAD]"] = max_frequency + 4
    dict["[UNK]"] = max_frequency + 3
    dict["[CLS]"] = max_frequency + 2
    dict["[SEP]"] = max_frequency + 1

    words = sorted(dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

    fout = open(dict_file, 'w')
    for word, frequency in words:
        fout.write(word + '\n')

    fout.close()


def main():
    """
    main
    """
    if len(sys.argv) < 3:
        print("Usage: " + sys.argv[0] + " corpus_file dict_file")
        exit()

    build_dict(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
