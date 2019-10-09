#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: conversation_strategy.py
"""

from __future__ import print_function

import sys

sys.path.append("../")
import network
from tools.convert_conversation_corpus_to_model_text import preprocessing_for_one_conversation

reload(sys)
sys.setdefaultencoding('utf8')


def load():
    """
    load model
    """
    return network.load()


def predict(model, text):
    """
    predict
    """
    model_text, topic_dict = \
        preprocessing_for_one_conversation(text.strip(), topic_generalization=True)

    if isinstance(model_text, unicode):
        model_text = model_text.encode('utf-8')

    response = network.predict(model, model_text)

    topic_list = sorted(topic_dict.items(), key=lambda item: len(item[1]), reverse=True)
    for key, value in topic_list:
        response = response.replace(key, value)

    return response


def main():
    """
    main
    """
    generator = load()
    for line in sys.stdin:
        response = predict(generator, line.strip())
        print(response)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
