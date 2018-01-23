#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
"""
File: test.py
Author: baidu(baidu@baidu.com)
Date: 2018/01/12 11:41:37
"""
import cPickle
#with open("./output/item_freq.pkl") as f:
with open("./data/nid_dict.pkl") as f:
    item_freq = cPickle.load(f)
print item_freq
