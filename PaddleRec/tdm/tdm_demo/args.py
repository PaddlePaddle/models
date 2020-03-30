# -*- coding=utf-8 -*-
"""
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import six
import argparse


def str2bool(v):
    """
    str2bool
    """
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    """
    ArgumentGroup
    """

    def __init__(self, parser, title, des):
        """
        init
        """
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        """
        add_arg
        """
        type = str2bool if type == bool else type
        # if type == list: # by dwk
        #     self._group.add_argument("--" + name, nargs='+', type=int)
        # else:
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)


def parse_args():
    """
    parse_args
    """
    # global
    parser = argparse.ArgumentParser("main")
    main_g = ArgumentGroup(parser, "main", "global conf")
    main_g.add_arg("random_seed", int, 0, "random_seed")
    main_g.add_arg("cpu_num", int, 1, "cpu_num")
    main_g.add_arg("is_local", bool, False,
                   "whether to perform local training")
    main_g.add_arg("is_cloud", bool, False, "")
    main_g.add_arg("is_test", bool, False, "")
    main_g.add_arg("sync_mode", str, "async", "distributed traing mode")
    main_g.add_arg("need_trace", bool, False, "")
    main_g.add_arg("need_detail", bool, False, "")

    # model
    model_g = ArgumentGroup(
        parser, "model", "options to init, resume and save model.")
    model_g.add_arg("epoch_num", int, 3, "number of epochs for train")
    model_g.add_arg("batch_size", int, 16, "batch size for train")
    model_g.add_arg("learning_rate", float, 5e-5,
                    "learning rate for global training")

    model_g.add_arg("layer_size", int, 4, "layer size")
    model_g.add_arg("node_nums", int, 26, "tree node nums")
    model_g.add_arg("node_emb_size", int, 64, "node embedding size")
    model_g.add_arg("query_emb_size", int, 768, "input query embedding size")
    model_g.add_arg("neg_sampling_list", list, [
                    1, 2, 3, 4], "nce sample nums at every layer")
    model_g.add_arg("layer_node_num_list", list, [
                    2, 4, 7, 12], "node nums at every layer")
    model_g.add_arg("leaf_node_num", int, 13, "leaf node nums")

    # for infer
    model_g.add_arg("child_nums", int, 2, "child node of ancestor node")
    model_g.add_arg("topK", int, 2, "best recall result nums")

    model_g = ArgumentGroup(
        parser, "path", "files path of data & model.")
    model_g.add_arg("train_files_path", str, "./data/train", "train data path")
    model_g.add_arg("test_files_path", str, "./data/test", "test data path")
    model_g.add_arg("model_files_path", str, "./models", "model data path")

    # build tree and warm up
    model_g.add_arg("build_tree_init_path", str,
                    "./data/gen_tree/demo_fake_input.txt", "build tree embedding path")
    model_g.add_arg("warm-up", bool, False,
                    "warm up, builing new tree.")
    model_g.add_arg("rebuild_tree_per_epochs", int, -1,
                    "re-build tree per epochs, -1 means don't re-building")

    model_g.add_arg("tree_info_init_path", str,
                    "./thirdparty/tree_info.txt", "embedding file path")
    model_g.add_arg("tree_travel_init_path", str,
                    "./thirdparty/travel_list.txt", "TDM tree travel file path")
    model_g.add_arg("tree_layer_init_path", str,
                    "./thirdparty/layer_list.txt", "TDM tree layer file path")
    model_g.add_arg("tree_id2item_init_path", str,
                    "./thirdparty/id2item.json", "item_id to item(feasign) mapping file path")
    model_g.add_arg("load_model", bool, False,
                    "whether load model(paddle persistables model)")
    model_g.add_arg("save_init_model", bool, False,
                    "whether save init model(paddle persistables model)")
    model_g.add_arg("init_model_files_path", str, "./models/init_model",
                    "init model params by paddle model files for training")

    args = parser.parse_args()
    return args


def print_arguments(args):
    """
    print arguments
    """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')
