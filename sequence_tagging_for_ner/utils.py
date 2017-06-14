#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import re
import argparse
import numpy as np
from collections import defaultdict

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)


def get_embedding(emb_file='data/wordVectors.txt'):
    """
    Get the trained word vector.
    """
    return np.loadtxt(emb_file, dtype=float)


def load_dict(dict_path):
    return dict((line.strip().split("\t")[0], idx)
                for idx, line in enumerate(open(dict_path, "r").readlines()))


def load_reverse_dict(dict_path):
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r").readlines()))
