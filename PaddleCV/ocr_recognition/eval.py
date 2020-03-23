#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from utility import add_arguments, print_arguments, to_lodtensor, get_ctc_feeder_data, get_attention_feeder_data
from utility import check_gpu
from attention_model import attention_eval
from crnn_ctc_model import ctc_eval
import data_reader
import argparse
import functools
import os

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model',    str,   "crnn_ctc",           "Which type of network to be used. 'crnn_ctc' or 'attention'")
add_arg('model_path',         str,  "",   "The model path to be used for inference.")
add_arg('input_images_dir',   str,  None,   "The directory of images.")
add_arg('input_images_list',  str,  None,   "The list file of images.")
add_arg('use_gpu',            bool,  True,      "Whether use GPU to eval.")
# yapf: enable


def evaluate(args):
    """OCR inference"""

    if args.model == "crnn_ctc":
        eval = ctc_eval
        get_feeder_data = get_ctc_feeder_data
    else:
        eval = attention_eval
        get_feeder_data = get_attention_feeder_data

    num_classes = data_reader.num_classes()
    data_shape = data_reader.data_shape()
    # define network
    evaluator, cost = eval(
        data_shape, num_classes, use_cudnn=True if args.use_gpu else False)

    # data reader
    test_reader = data_reader.test(
        test_images_dir=args.input_images_dir,
        test_list_file=args.input_images_list,
        model=args.model)

    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load init model
    model_dir = args.model_path
    if os.path.isdir(args.model_path):
        raise Exception("{} should not be a directory".format(args.model_path))
    fluid.load(
        program=fluid.default_main_program(),
        model_path=model_dir,
        executor=exe,
        var_list=fluid.io.get_program_parameter(fluid.default_main_program()))
    print("Init model from: %s." % args.model_path)

    evaluator.reset(exe)
    count = 0
    for data in test_reader():
        count += 1
        exe.run(fluid.default_main_program(), feed=get_feeder_data(data, place))
    avg_distance, avg_seq_error = evaluator.eval(exe)
    print("Read %d samples; avg_distance: %s; avg_seq_error: %s" %
          (count, avg_distance, avg_seq_error))


def main():
    args = parser.parse_args()
    print_arguments(args)
    check_gpu(args.use_gpu)
    evaluate(args)


if __name__ == "__main__":
    main()
