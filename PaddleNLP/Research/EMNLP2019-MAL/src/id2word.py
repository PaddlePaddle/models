#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: word2id.py
Author: bitianchi(bitianchi@baidu.com)
Date: 2019/05/27 14:42:54
"""

import sys

id2word = {}
ln = sys.stdin

def load_vocab(file_path):
    start_index = 0
    f = open(file_path, 'r')

    for line in f:
        line = line.strip()
        id2word[start_index] = line
        start_index += 1
    f.close()

if __name__=="__main__":
    load_vocab(sys.argv[1])
    while True:
        line = ln.readline().strip()
        if not line:
            break

        split_res = line.split(" ")
        output_str = ""
        for item in split_res:
            output_str += id2word[int(item.strip())]
            output_str += " "
        output_str = output_str.strip()
        print output_str

