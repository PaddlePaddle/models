#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: convert_session_to_sample.py
"""

from __future__ import print_function

import sys
import json
import collections

reload(sys)
sys.setdefaultencoding('utf8')


def convert_session_to_sample(session_file, sample_file):
    """
    convert_session_to_sample
    """
    fout = open(sample_file, 'w')
    with open(session_file, 'r') as f:
        for i, line in enumerate(f):
            session = json.loads(line.strip(), encoding="utf-8", \
                                      object_pairs_hook=collections.OrderedDict)
            conversation = session["conversation"]

            for j in range(0, len(conversation), 2):
                sample = collections.OrderedDict()
                sample["goal"] = session["goal"]
                sample["knowledge"] = session["knowledge"]
                sample["history"] = conversation[:j]
                sample["response"] = conversation[j]

                sample = json.dumps(sample, ensure_ascii=False, encoding="utf-8")

                fout.write(sample + "\n")

    fout.close()


def main():
    """
    main
    """
    convert_session_to_sample(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
