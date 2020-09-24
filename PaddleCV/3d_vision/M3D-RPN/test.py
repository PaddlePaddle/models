#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
"""test"""

import os
import sys
import argparse
import ast
import logging
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from models import *
from easydict import EasyDict as edict
from lib.rpn_util import *

sys.path.append(os.getcwd())
import lib.core as core
from lib.util import *
import pdb

import paddle
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid import framework

logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    """parse"""
    parser = argparse.ArgumentParser("M3D-RPN train script")
    parser.add_argument("--conf_path", type=str, default='', help="config.pkl")
    parser.add_argument(
        '--weights_path', type=str, default='', help='weights save path')

    parser.add_argument(
        '--backbone',
        type=str,
        default='DenseNet121',
        help='backbone model to train, default DenseNet121')

    parser.add_argument(
        '--data_dir', type=str, default='dataset', help='dataset directory')

    args = parser.parse_args()
    return args


def test():
    """main train"""
    args = parse_args()
    # load config
    conf = edict(pickle_read(args.conf_path))
    conf.pretrained = None

    results_path = os.path.join('output', 'tmp_results', 'data')
    # make directory
    mkdir_if_missing(results_path, delete_if_exist=True)

    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        # training network
        src_path = os.path.join('.', 'models', conf.model + '.py')
        train_model = absolute_import(src_path)
        train_model = train_model.build(conf, args.backbone, 'train')
        train_model.eval()
        train_model.phase = "eval"
        Already_trained, _ = fluid.load_dygraph(args.weights_path)
        print("loaded model from ", args.weights_path)
        train_model.set_dict(Already_trained)  #, use_structured_name=True)
        print("start evaluation...")
        test_kitti_3d(conf.dataset_test, train_model, conf, results_path,
                      args.data_dir)
    print("Evaluation Finished!")


if __name__ == '__main__':

    test()
