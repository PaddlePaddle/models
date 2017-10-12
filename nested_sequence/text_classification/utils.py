#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import argparse
from collections import defaultdict

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def parse_train_cmd():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle text classification demo")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=False,
        help=("path of training dataset (default: None). "
              "if this parameter is not set, "
              "imdb dataset will be used."),
        default=None)
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=False,
        help=("path of testing dataset (default: None). "
              "if this parameter is not set, "
              "imdb dataset will be used."),
        default=None)
    parser.add_argument(
        "--word_dict",
        type=str,
        required=False,
        help=("path of word dictionary (default: None)."
              "if this parameter is not set, imdb dataset will be used."
              "if this parameter is set, but the file does not exist, "
              "word dictionay will be built from "
              "the training data automatically."),
        default=None)
    parser.add_argument(
        "--class_num",
        type=int,
        required=False,
        help=("class number."),
        default=2)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="the number of training examples in one forward/backward pass")
    parser.add_argument(
        "--num_passes", type=int, default=10, help="number of passes to train")
    parser.add_argument(
        "--model_save_dir",
        type=str,
        required=False,
        help=("path to save the trained models."),
        default="models")

    return parser.parse_args()
