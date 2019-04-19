"""
Arguments for configuration
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import argparse
import io
import sys
import random
import numpy as np
import os

import paddle
import paddle.fluid as fluid


def str2bool(v):
    """
    String to Boolean
    """
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    """
    Argument Class
    """
    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        """
        Add argument
        """
        type = str2bool if type == bool else type
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)


def print_arguments(args):
    """
    Print Arguments
    """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def init_checkpoint(exe, init_checkpoint_path, main_program):
    """
    Init CheckPoint
    """
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path

    def existed_persitables(var):
        """
        If existed presitabels
        """
        if not fluid.io.is_persistable(var):
            return False
        return os.path.exists(os.path.join(init_checkpoint_path, var.name))

    fluid.io.load_vars(
        exe,
        init_checkpoint_path,
        main_program=main_program,
        predicate=existed_persitables)
    print("Load model from {}".format(init_checkpoint_path))

    
def data_reader(file_path, word_dict, num_examples, phrase, epoch):
    """
    Convert word sequence into slot
    """
    unk_id = len(word_dict)
    all_data = []
    with io.open(file_path, "r", encoding='utf8') as fin:
        for line in fin:
            if line.startswith('text_a'):
                continue
            cols = line.strip().split("\t")
            if len(cols) != 2:
                sys.stderr.write("[NOTICE] Error Format Line!")
                continue
            label = int(cols[1])
            wids = [word_dict[x] if x in word_dict else unk_id
                    for x in cols[0].split(" ")]
            all_data.append((wids, label))

    if phrase == "train":
        random.shuffle(all_data)

    num_examples[phrase] = len(all_data)
        
    def reader():
        """
        Reader Function
        """
        for epoch_index in range(epoch):
            for doc, label in all_data:
                yield doc, label
    return reader

def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    with io.open(file_path, 'r', encoding='utf8') as f:
        wid = 0
        for line in f:
            if line.strip() not in vocab:
                vocab[line.strip()] = wid
                wid += 1
    vocab["<unk>"] = len(vocab)
    return vocab


def init_pretraining_params(exe,
                            pretraining_params_path,
                            main_program,
                            use_fp16=False):
    """load params of pretrained model, NOT including moment, learning_rate"""
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path

    def _existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(pretraining_params_path, var.name))

    fluid.io.load_vars(
        exe,
        pretraining_params_path,
        main_program=main_program,
        predicate=_existed_params)
    print("Load pretraining parameters from {}.".format(
        pretraining_params_path))
