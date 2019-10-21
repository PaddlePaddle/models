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
import time

import paddle
import paddle.fluid as fluid
import imagenet_reader as reader
sys.path.append("..")
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
# yapf: disable
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_gpu',          bool, False,                 "Whether to use GPU or not.")
add_arg('model_path', str,  "./pruning/checkpoints/resnet50/2/eval_model/",                 "Whether to use pretrained model.")
add_arg('model_name', str,  "__model__.infer",  "inference model filename")
add_arg('params_name', str, "__params__", "inference model params filename")
# yapf: enable

def infer(args):
    # parameters from arguments

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    test_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(args.model_path,
                                      exe,
                                      model_filename=args.model_name,
                                      params_filename=args.params_name)
    test_reader = paddle.batch(reader.test(), batch_size=1)
    feeder = fluid.DataFeeder(place=place, feed_list=feed_target_names, program=test_program)

    results=[]
    #for infer time, if you don't need, please change infer_time to False
    infer_time = True
    for batch_id, data in enumerate(test_reader()):
        # for infer time
        if infer_time:
            warmup_times = 10
            repeats_time = 30
            feed_data = feeder.feed(data)
            for i in range(warmup_times):
                exe.run(test_program,
                        feed=feed_data,
                        fetch_list=fetch_targets)
            start_time = time.time()
            for i in range(repeats_time):
                exe.run(test_program,
                        feed=feed_data,
                        fetch_list=fetch_targets)
            print("infer time: {} ms/sample".format((time.time()-start_time) * 1000 / repeats_time))
            infer_time = False
        # top1_acc, top5_acc
        result = exe.run(test_program,
                          feed=feeder.feed(data),
                          fetch_list=fetch_targets)
        result = np.array(result[0])
        print(result.argsort(axis=1)[:,-1:][::-1])
    sys.stdout.flush()

def main():
    args = parser.parse_args()
    print_arguments(args)
    infer(args)

if __name__ == '__main__':
    main()
