#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
"""
Contains common utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import distutils.util
import numpy as np
import six
import ast
from collections import deque
import paddle.fluid as fluid
import argparse
import functools
from config import *


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self):
        self.loss_sum = 0.0
        self.iter_cnt = 0

    def add_value(self, value):
        self.loss_sum += np.mean(value)
        self.iter_cnt += 1

    def get_mean_value(self):
        return self.loss_sum / self.iter_cnt


def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=True in paddlepaddle
    cpu version.
    """
    err = "Config use_gpu cannot be set as True while you are " \
          "using paddlepaddle cpu version ! \nPlease try: \n" \
          "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
          "\t2. Set --use_gpu=False to run model on CPU"

    try:
        if use_gpu and not fluid.is_compiled_with_cuda():
            print(err)
            sys.exit(1)
    except Exception as e:
        pass


def parse_args():
    """return all args
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    # yapf: disable
    # ENV
    add_arg('use_gpu',          bool,   True,      "Whether use GPU.")
    add_arg('model_save_dir',   str,    'checkpoints',     "The path to save model.")
    add_arg('pretrain',         str,    'weights/darknet53', "The pretrain model path.")
    add_arg('finetune',         str,    False, "The finetune model path.")
    add_arg('weights',          str,    'weights/yolov3', "The weights path.")
    add_arg('dataset',          str,    'coco2017',  "Dataset: coco2014, coco2017.")
    add_arg('class_num',        int,    80,          "Class number.")
    add_arg('data_dir',         str,    'dataset/coco',        "The data root path.")
    add_arg('start_iter',       int,    0,      "Start iteration.")
    add_arg('use_multiprocess_reader', bool,   True,   "whether use multiprocess reader.")
    add_arg('use_data_parallel', ast.literal_eval, False, "the flag indicating whether to use data parallel model to train the model")
    #SOLVER
    add_arg('batch_size',       int,    8,      "Mini-batch size per device.")
    add_arg('learning_rate',    float,  0.001,  "Learning rate.")
    add_arg('max_iter',         int,    500200, "Iter number.")
    add_arg('snapshot_iter',    int,    2000,   "Save model every snapshot stride.")
    add_arg('label_smooth',     bool,   True,   "Use label smooth in class label.")
    add_arg('no_mixup_iter',    int,    40000,  "Disable mixup in last N iter.")
    # TRAIN TEST INFER
    add_arg('input_size',       int,    608,    "Image input size of YOLOv3.")
    add_arg('random_shape',     bool,   True,   "Resize to random shape for train reader.")
    add_arg('valid_thresh',     float,  0.005,  "Valid confidence score for NMS.")
    add_arg('nms_thresh',       float,  0.45,   "NMS threshold.")
    add_arg('nms_topk',         int,    400,    "The number of boxes to perform NMS.")
    add_arg('nms_posk',         int,    100,    "The number of boxes of NMS output.")
    add_arg('debug',            bool,   False,  "Debug mode")
    # SINGLE EVAL AND DRAW
    add_arg('image_path',       str,   'image',
            "The image path used to inference and visualize.")
    add_arg('image_name',       str,    None,
            "The single image used to inference and visualize. None to inference all images in image_path")
    add_arg('draw_thresh',      float,  0.5,
            "Confidence score threshold to draw prediction box in image in debug mode")
    add_arg('enable_ce',        bool,  False,                "If set True, enable continuous evaluation job.")
    # yapf: enable
    args = parser.parse_args()
    file_name = sys.argv[0]
    merge_cfg_from_args(args)
    return args
