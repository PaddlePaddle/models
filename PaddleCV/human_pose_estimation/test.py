# Copyright (c) 2018-present, Baidu, Inc.
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
##############################################################################
"""Functions for inference."""

import sys
import argparse
import functools
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers

from lib import pose_resnet
from utils.transforms import flip_back
from utils.utility import *

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('batch_size',       int,   32,                  "Minibatch size.")
add_arg('dataset',          str,   'mpii',              "Dataset")
add_arg('use_gpu',          bool,  True,                "Whether to use GPU or not.")
add_arg('kp_dim',           int,   16,                  "Class number.")
add_arg('checkpoint',       str,   None,                "Whether to resume checkpoint.")
add_arg('flip_test',        bool,  True,                "Flip test")
add_arg('shift_heatmap',    bool,  True,                "Shift heatmap")
# yapf: enable


def print_immediately(s):
    print(s)
    sys.stdout.flush()


def test(args):
    if args.dataset == 'coco':
        import lib.coco_reader as reader
        IMAGE_SIZE = [288, 384]
        FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                      [13, 14], [15, 16]]
        args.kp_dim = 17
    elif args.dataset == 'mpii':
        import lib.mpii_reader as reader
        IMAGE_SIZE = [384, 384]
        FLIP_PAIRS = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        args.kp_dim = 16
    else:
        raise ValueError('The dataset {} is not supported yet.'.format(
            args.dataset))

    print_arguments(args)

    # Image and target
    image = layers.data(
        name='image', shape=[3, IMAGE_SIZE[1], IMAGE_SIZE[0]], dtype='float32')
    file_id = layers.data(name='file_id', shape=[1, ], dtype='int')

    # Build model
    model = pose_resnet.ResNet(layers=50, kps_num=args.kp_dim, test_mode=True)

    # Output
    output = model.net(input=image, target=None, target_weight=None)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if args.checkpoint is not None:
        fluid.io.load_persistables(exe, args.checkpoint)

    # Dataloader
    test_reader = paddle.batch(reader.test(), batch_size=args.batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, file_id])

    test_exe = fluid.ParallelExecutor(
        use_cuda=True if args.use_gpu else False,
        main_program=fluid.default_main_program().clone(for_test=True),
        loss_name=None)

    fetch_list = [image.name, output.name]

    for batch_id, data in enumerate(test_reader()):
        print_immediately("Processing batch #%d" % batch_id)
        num_images = len(data)

        file_ids = []
        for i in range(num_images):
            file_ids.append(data[i][1])

        input_image, out_heatmaps = test_exe.run(fetch_list=fetch_list,
                                                 feed=feeder.feed(data))

        if args.flip_test:
            # Flip all the images in a same batch
            data_fliped = []
            for i in range(num_images):
                data_fliped.append((data[i][0][:, :, ::-1], data[i][1]))

            # Inference again
            _, output_flipped = test_exe.run(fetch_list=fetch_list,
                                             feed=feeder.feed(data_fliped))

            # Flip back
            output_flipped = flip_back(output_flipped, FLIP_PAIRS)

            # Feature is not aligned, shift flipped heatmap for higher accuracy
            if args.shift_heatmap:
                output_flipped[:, :, :, 1:] = \
                        output_flipped.copy()[:, :, :, 0:-1]

            # Aggregate
            out_heatmaps = (out_heatmaps + output_flipped) * 0.5
            save_predict_results(
                input_image, out_heatmaps, file_ids, fold_name='results')


if __name__ == '__main__':
    args = parser.parse_args()
    check_cuda(args.use_gpu)
    test(args)
