# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" 
Copy-paste from PaddleSeg with minor modifications.
https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/val.py
"""

import argparse
import os

import paddle

from smoke.cvlibs import manager, Config
from smoke.core import evaluate
from smoke.utils import logger, load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # params of evaluate
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, required=True, type=str)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for evaluation',
        type=str,
        default=None)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)
    parser.add_argument(
        '--output_dir',
        dest='output_dir',
        help='The directory for saving the evaluation results',
        type=str,
        default='./output')
   

    return parser.parse_args()


def main(args):
    
    paddle.set_device("gpu")

    cfg = Config(args.cfg)
    val_dataset = cfg.val_dataset
    if val_dataset is None:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )
    elif len(val_dataset) == 0:
        raise ValueError(
            'The length of val_dataset is 0. Please check if your dataset is valid'
        )

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    if args.model_path:
        load_pretrained_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')

    evaluate(
        model,
        val_dataset,
        num_workers=args.num_workers,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)
