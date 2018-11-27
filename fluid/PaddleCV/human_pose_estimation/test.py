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

import os
import argparse
import functools
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers

from tqdm import tqdm
from lib import pose_resnet
from utils.transforms import flip_back
from utils.utility import *

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('batch_size',       int,   32,                  "Minibatch size.")
add_arg('dataset',          str,   'mpii',              "Dataset")
add_arg('use_gpu',          bool,  True,                "Whether to use GPU or not.")
add_arg('num_epochs',       int,   140,                 "Number of epochs.")
add_arg('total_images',     int,   144406,              "Training image number.")
add_arg('kp_dim',           int,   16,                  "Class number.")
add_arg('model_save_dir',   str,   "output",            "Model save directory")
add_arg('with_mem_opt',     bool,  True,               "Whether to use memory optimization or not.")
add_arg('pretrained_model', str,   None,                "Whether to use pretrained model.")
add_arg('checkpoint',       str,   None,                "Whether to resume checkpoint.")
add_arg('lr',               float, 0.001,               "Set learning rate.")
add_arg('lr_strategy',      str,   "piecewise_decay",   "Set the learning rate decay strategy.")
add_arg('flip_test',        bool,  True,                "Flip test")
add_arg('shift_heatmap',    bool,  True,                "Shift heatmap")
add_arg('post_process',     bool,  False,               "post process")
# yapf: enable

FLIP_PAIRS = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]

def test(args):
    if args.dataset == 'coco':
        import lib.coco_reader as reader
        IMAGE_SIZE = [288, 384]
        # HEATMAP_SIZE = [72, 96]
        FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        args.kp_dim = 17
        args.total_images = 144406 # 149813
    elif args.dataset == 'mpii':
        import lib.mpii_reader as reader
        IMAGE_SIZE = [384, 384]
        # HEATMAP_SIZE = [96, 96]
        FLIP_PAIRS = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        args.kp_dim = 16
        args.total_images = 2958 # validation
    else:
        raise ValueError('The dataset {} is not supported yet.'.format(args.dataset))

    print_arguments(args)

    # Image and target
    image = layers.data(name='image', shape=[3, IMAGE_SIZE[1], IMAGE_SIZE[0]], dtype='float32')
    file_id = layers.data(name='file_id', shape=[1,], dtype='int')

    # Build model
    model = pose_resnet.ResNet(layers=50, kps_num=args.kp_dim, test_mode=True)

    # Output
    output = model.net(input=image, target=None, target_weight=None)

    # Parameters from model and arguments
    params = {}
    params["total_images"] = args.total_images
    params["lr"] = args.lr
    params["num_epochs"] = args.num_epochs
    params["learning_strategy"] = {}
    params["learning_strategy"]["batch_size"] = args.batch_size
    params["learning_strategy"]["name"] = args.lr_strategy

    if args.with_mem_opt:
        fluid.memory_optimize(fluid.default_main_program(),
                              skip_opt_set=[output.name])

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    args.pretrained_model = './pretrained/resnet_50/115'
    if args.pretrained_model:
        def if_exist(var):
            exist_flag = os.path.exists(os.path.join(args.pretrained_model, var.name))
            return exist_flag
        fluid.io.load_vars(exe, args.pretrained_model, predicate=if_exist)

    if args.checkpoint is not None:
        fluid.io.load_persistables(exe, args.checkpoint)

    # Dataloader
    test_reader = paddle.batch(reader.test(), batch_size=args.batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, file_id])

    test_exe = fluid.ParallelExecutor(
            use_cuda=True if args.use_gpu else False,
            main_program=fluid.default_main_program().clone(for_test=False),
            loss_name=None)

    fetch_list = [image.name, output.name]

    for batch_id, data in tqdm(enumerate(test_reader())):
        num_images = len(data)

        file_ids = []
        for i in range(num_images):
            file_ids.append(data[i][1])

        input_image, out_heatmaps  = test_exe.run(
                fetch_list=fetch_list,
                feed=feeder.feed(data))

        if args.flip_test:
            # Flip all the images in a same batch
            data_fliped = []
            for i in range(num_images):
                data_fliped.append((
                            data[i][0][:, :, ::-1],
                            data[i][1]))

            # Inference again
            _, output_flipped = test_exe.run(
                    fetch_list=fetch_list,
                    feed=feeder.feed(data_fliped))

            # Flip back
            output_flipped = flip_back(output_flipped, FLIP_PAIRS)

            # Feature is not aligned, shift flipped heatmap for higher accuracy
            if args.shift_heatmap:
                output_flipped[:, :, :, 1:] = \
                        output_flipped.copy()[:, :, :, 0:-1]

            # Aggregate
            out_heatmaps = (out_heatmaps + output_flipped) * 0.5
            save_predict_results(input_image, out_heatmaps, file_ids, fold_name='results')

if __name__ == '__main__':
    args = parser.parse_args()
    test(args)
