"""
EmoTect utilities.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import six
import sys
import random
import argparse

import paddle
import paddle.fluid as fluid
import numpy as np

def str2bool(value):
    """
    String to Boolean
    """
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return value.lower() in ("true", "t", "1")


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


def data_reader(file_path, word_dict, num_examples, phrase, epoch=1):
    """
    Convert word sequence into slot
    """
    unk_id = len(word_dict)
    all_data = []
    with io.open(file_path, "r", encoding='utf8') as fin:
        for line in fin:
            if line.startswith("label"):
                continue
            if phrase == "infer":
                cols = line.strip().split("\t")
                if len(cols) != 1:
                    query = cols[-1]
                wids = [word_dict[x] if x in word_dict else unk_id
                        for x in query.strip().split(" ")]
                all_data.append((wids,))
            else:
                cols = line.strip().split("\t")
                if len(cols) != 2:
                    sys.stderr.write("[NOTICE] Error Format Line!")
                    continue
                label = int(cols[0])
                wids = [word_dict[x] if x in word_dict else unk_id
                        for x in cols[1].split(" ")]
                all_data.append((wids, label))
    num_examples[phrase] = len(all_data)

    if phrase == "infer":
        def reader():
            """
            Infer reader function
            """
            for wids in all_data:
                yield wids
        return reader

    def reader():
        """
        Reader function
        """
        for idx in range(epoch):
            if phrase == "train":
                random.shuffle(all_data)
            for wids, label in all_data:
                yield wids, label
    return reader


def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    with io.open(file_path, 'r', encoding='utf8') as fin:
        wid = 0
        for line in fin:
            if line.strip() not in vocab:
                vocab[line.strip()] = wid
                wid += 1
    vocab["<unk>"] = len(vocab)
    return vocab
