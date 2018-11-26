#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import re
import argparse
import numpy as np
from collections import defaultdict

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def get_embedding(emb_file='data/wordVectors.txt'):
    """
    Get the trained word vector.
    """
    return np.loadtxt(emb_file, dtype=float)


def load_dict(dict_path):
    """
    Load the word dictionary from the given file.
    Each line of the given file is a word, which can include multiple columns
    seperated by tab.

    This function takes the first column (columns in a line are seperated by
    tab) as key and takes line number of a line as the key (index of the word
    in the dictionary).
    """

    return dict((line.strip().split("\t")[0], idx)
                for idx, line in enumerate(open(dict_path, "r").readlines()))


def load_reverse_dict(dict_path):
    """
    Load the word dictionary from the given file.
    Each line of the given file is a word, which can include multiple columns
    seperated by tab.

    This function takes line number of a line as the key (index of the word in
    the dictionary) and the first column (columns in a line are seperated by
    tab) as the value.
    """
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r").readlines()))
