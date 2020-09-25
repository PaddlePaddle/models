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

import argparse
import logging
import numpy as np
import os
import paddle
import paddle.fluid as fluid
import reader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle DIN example")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="path of model parameters")
    parser.add_argument(
        '--test_path',
        type=str,
        default='data/paddle_test.txt.bak',
        help='dir of test file')
    parser.add_argument(
        '--use_cuda', type=int, default=0, help='whether to use gpu')

    return parser.parse_args()


def calc_auc(raw_arr):
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d: d[2])
    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0]  # noclick
        tp2 += record[1]  # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2
    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5
    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None


def infer():
    args = parse_args()
    model_path = args.model_path
    use_cuda = True if args.use_cuda else False
    data_reader, _ = reader.prepare_reader(args.test_path, 32 * 16)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inference_scope = fluid.Scope()

    exe = fluid.Executor(place)

    #with fluid.scope_guard(inference_scope):
    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(model_path, exe)

    loader = fluid.io.DataLoader.from_generator(
        feed_list=[
            inference_program.block(0).var(e) for e in feed_target_names
        ],
        capacity=10000,
        iterable=True)
    loader.set_sample_list_generator(data_reader, places=place)

    loss_sum = 0.0
    score = []
    count = 0
    for data in loader():
        res = exe.run(inference_program, feed=data, fetch_list=fetch_targets)
        loss_sum += res[0]
        label_data = list(np.array(data[0]["label"]))
        for i in range(len(label_data)):
            if label_data[i] > 0.5:
                score.append([0, 1, res[1][i]])
            else:
                score.append([1, 0, res[1][i]])
        count += 1
    auc = calc_auc(score)
    logger.info("TEST --> loss: {}, auc: {}".format(loss_sum / count, auc))


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    infer()
