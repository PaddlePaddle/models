#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import os
import sys
import numpy as np
import argparse
import functools
import logging

import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid import core
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization import QuantizationFreezePass
from paddle.fluid.contrib.slim.quantization import ConvertToInt8Pass
from paddle.fluid.contrib.slim.quantization import TransformForMobilePass
sys.path.append("..")
import imagenet_reader as reader
sys.path.append("../../")
from utility import add_arguments, print_arguments

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
# yapf: disable
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_gpu',          bool, True,                 "Whether to use GPU or not.")
add_arg('model_path', str,  "./pruning/checkpoints/resnet50/2/eval_model/",                 "Whether to use pretrained model.")
add_arg('save_path', str, './output',   'Path to save inference model')
add_arg('weight_quant_type', str, 'abs_max', 'quantization type for weight')
# yapf: enable

def eval(args):
    # parameters from arguments

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    val_program, feed_names, fetch_targets = fluid.io.load_inference_model(args.model_path,
                                      exe,
                                      model_filename="__model__",
                                      params_filename="__params__")
    val_reader = paddle.batch(reader.val(), batch_size=128)
    feeder = fluid.DataFeeder(place=place, feed_list=feed_names, program=val_program)

    results=[]
    for batch_id, data in enumerate(val_reader()):

        # top1_acc, top5_acc
        result = exe.run(val_program,
                          feed=feeder.feed(data),
                          fetch_list=fetch_targets)
        result = [np.mean(r) for r in result]
        results.append(result)
    result = np.mean(np.array(results), axis=0)
    print("top1_acc/top5_acc= {}".format(result))
    sys.stdout.flush()
    _logger.info("freeze the graph for inference")
    test_graph = IrGraph(core.Graph(val_program.desc), for_test=True)

    freeze_pass = QuantizationFreezePass(
            scope=fluid.global_scope(),
            place=place,
            weight_quantize_type=args.weight_quant_type)
    freeze_pass.apply(test_graph)
    server_program = test_graph.to_program()
    fluid.io.save_inference_model(
            dirname=os.path.join(args.save_path, 'float'),
            feeded_var_names=feed_names,
            target_vars=fetch_targets,
            executor=exe,
            main_program=server_program,
            model_filename='model',
            params_filename='weights')

    _logger.info("convert the weights into int8 type")
    convert_int8_pass = ConvertToInt8Pass(
            scope=fluid.global_scope(),
            place=place)
    convert_int8_pass.apply(test_graph)
    server_int8_program = test_graph.to_program()
    fluid.io.save_inference_model(
            dirname=os.path.join(args.save_path, 'int8'),
            feeded_var_names=feed_names,
            target_vars=fetch_targets,
            executor=exe,
            main_program=server_int8_program,
            model_filename='model',
            params_filename='weights')

    _logger.info("convert the freezed pass to paddle-lite execution")
    mobile_pass = TransformForMobilePass()
    mobile_pass.apply(test_graph)
    mobile_program = test_graph.to_program()
    fluid.io.save_inference_model(
            dirname=os.path.join(args.save_path, 'mobile'),
            feeded_var_names=feed_names,
            target_vars=fetch_targets,
            executor=exe,
            main_program=mobile_program,
            model_filename='model',
            params_filename='weights')

def main():
    args = parser.parse_args()
    print_arguments(args)
    eval(args)

if __name__ == '__main__':
    main()
