# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec
import os
import sys
import numpy as np


# parse args
def get_args(add_help=True):
    """get_args

    Parse all args using argparse lib

    Args:
        add_help: Whether to add -h option on args

    Returns:
        An object which contains many parameters used for inference.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Args', add_help=add_help)
    args = parser.parse_args()
    return args


def build_model(args):
    """build_model

    Build your own model.

    Args:
        args: Parameters generated using argparser.

    Returns:
        A model whose type is nn.Layer
    """
    pass


def export(args):
    """export

    export inference model using jit.save

    Args:
        args: Parameters generated using argparser.

    Returns: None
    """
    model = build_model(args)

    # decorate model with jit.save
    model = paddle.jit.to_static(
        model,
        input_spec=[
            InputSpec(
                shape=[None, 3, args.img_size, args.img_size], dtype='float32')
        ])
    # save inference model
    paddle.jit.save(model, os.path.join(args.save_inference_dir, "inference"))
    print(f"inference model is saved in {args.save_inference_dir}")


if __name__ == "__main__":
    args = get_args()
    export(args)
