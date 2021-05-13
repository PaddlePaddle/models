
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os

import paddle

from smoke.cvlibs import Config
from smoke.utils import load_pretrained_model

def parse_args():
    parser = argparse.ArgumentParser(description='Model Export')

    # params of evaluate
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", required=True, type=str)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for evaluation',
        type=str,
        required=True)
    parser.add_argument(
        '--output_dir',
        dest='output_dir',
        help='The directory saving inference params.',
        type=str,
        default="./deploy")
   

    return parser.parse_args()


def main(args):

    cfg = Config(args.cfg)
   
    model = cfg.model
    model.eval()

    load_pretrained_model(model, args.model_path)

    model = paddle.jit.to_static(model,
                             input_spec=[
                                 paddle.static.InputSpec(
                                     shape=[1, 3, None, None], dtype="float32",
                                 ),
                                 [
                                     paddle.static.InputSpec(
                                     shape=[1, 3, 3], dtype="float32"
                                     ),
                                     paddle.static.InputSpec(
                                     shape=[1, 2], dtype="float32"
                                     )
                                 ]
                             ]
                 )
    
    paddle.jit.save(model, os.path.join(args.output_dir, "inference"))

if __name__ == '__main__':
    args = parse_args()
    main(args)