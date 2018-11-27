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

"""Functions for validation."""

import os
import argparse
import functools
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers

from collections import OrderedDict
from scipy.io import loadmat, savemat
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
add_arg('post_process',     bool,  True,               "Post process")
# yapf: enable

def valid(args):
    if args.dataset == 'coco':
        import lib.coco_reader as reader
        IMAGE_SIZE = [288, 384]
        HEATMAP_SIZE = [72, 96]
        FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        args.kp_dim = 17
        args.total_images = 144406 # 149813
    elif args.dataset == 'mpii':
        import lib.mpii_reader as reader
        IMAGE_SIZE = [384, 384]
        HEATMAP_SIZE = [96, 96]
        FLIP_PAIRS = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        args.kp_dim = 16
        args.total_images = 2958 # validation
    else:
        raise ValueError('The dataset {} is not supported yet.'.format(args.dataset))

    print_arguments(args)

    # Image and target
    image = layers.data(name='image', shape=[3, IMAGE_SIZE[1], IMAGE_SIZE[0]], dtype='float32')
    target = layers.data(name='target', shape=[args.kp_dim, HEATMAP_SIZE[1], HEATMAP_SIZE[0]], dtype='float32')
    target_weight = layers.data(name='target_weight', shape=[args.kp_dim, 1], dtype='float32')
    center = layers.data(name='center', shape=[2,], dtype='float32')
    scale = layers.data(name='scale', shape=[2,], dtype='float32')
    score = layers.data(name='score', shape=[1,], dtype='float32')

    # Build model
    model = pose_resnet.ResNet(layers=50, kps_num=args.kp_dim)

    # Output
    loss, output = model.net(input=image, target=target, target_weight=target_weight)

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
                              skip_opt_set=[loss.name, output.name, target.name])

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
    valid_reader = paddle.batch(reader.valid(), batch_size=args.batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, target, target_weight, center, scale, score])

    valid_exe = fluid.ParallelExecutor(
            use_cuda=True if args.use_gpu else False,
            main_program=fluid.default_main_program().clone(for_test=False),
            loss_name=loss.name)

    fetch_list = [image.name, loss.name, output.name, target.name]

    # For validation
    acc = AverageMeter()
    idx = 0

    num_samples = args.total_images
    all_preds = np.zeros((num_samples, args.kp_dim, 3),
                        dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))

    for batch_id, data in enumerate(valid_reader()):
        num_images = len(data)

        centers = []
        scales = []
        scores = []
        for i in range(num_images):
            centers.append(data[i][3])
            scales.append(data[i][4])
            scores.append(data[i][5])

        input_image, loss, out_heatmaps, target_heatmaps = valid_exe.run(
                fetch_list=fetch_list,
                feed=feeder.feed(data))

        if args.flip_test:
            # Flip all the images in a same batch
            data_fliped = []
            for i in range(num_images):
                # Input, target, target_weight, c, s, score
                data_fliped.append((
                            # np.flip(input_image, 3)[i],
                            data[i][0][:, :, ::-1],
                            data[i][1],
                            data[i][2],
                            data[i][3],
                            data[i][4],
                            data[i][5]))

            # Inference again
            _, _, output_flipped, _ = valid_exe.run(
                    fetch_list=fetch_list,
                    feed=feeder.feed(data_fliped))

            # Flip back
            output_flipped = flip_back(output_flipped, FLIP_PAIRS)

            # Feature is not aligned, shift flipped heatmap for higher accuracy
            if args.shift_heatmap:
                output_flipped[:, :, :, 1:] = \
                        output_flipped.copy()[:, :, :, 0:-1]

            # Aggregate
            # out_heatmaps.shape: size[b, args.kp_dim, 96, 96]
            out_heatmaps = (out_heatmaps + output_flipped) * 0.5

        loss = np.mean(np.array(loss))

        # Accuracy
        _, avg_acc, cnt, pred = accuracy(out_heatmaps, target_heatmaps)
        acc.update(avg_acc, cnt)

        # Current center, scale, score
        centers = np.array(centers)
        scales = np.array(scales)
        scores = np.array(scores)

        preds, maxvals = get_final_preds(
            args, out_heatmaps, centers, scales)

        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        # Double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0:2] = centers[:, 0:2]
        all_boxes[idx:idx + num_images, 2:4] = scales[:, 0:2]
        all_boxes[idx:idx + num_images, 4] = np.prod(scales*200, 1)
        all_boxes[idx:idx + num_images, 5] = scores
        # image_path.extend(meta['image'])

        idx += num_images

        print('Epoch [{:4d}] '
              'Loss = {:.5f} '
              'Acc = {:.5f}'.format(batch_id, loss, acc.avg))

        if batch_id % 10 == 0:
            save_batch_heatmaps(input_image, out_heatmaps, file_name='visualization@val.jpg', normalize=True)

    # Evaluate
    args.DATAROOT = 'data/mpii'
    args.TEST_SET = 'valid'
    output_dir = ''
    filenames = []
    imgnums = []
    image_path = []
    name_values, perf_indicator = mpii_evaluate(
        args, all_preds, output_dir, all_boxes, image_path,
        filenames, imgnums)

    print_name_value(name_values, perf_indicator)

def mpii_evaluate(cfg, preds, output_dir, *args, **kwargs):
    # Convert 0-based index to 1-based index
    preds = preds[:, :, 0:2] + 1.0

    if output_dir:
        pred_file = os.path.join(output_dir, 'pred.mat')
        savemat(pred_file, mdict={'preds': preds})

    if 'test' in cfg.TEST_SET:
        return {'Null': 0.0}, 0.0

    SC_BIAS = 0.6
    threshold = 0.5

    gt_file = os.path.join(cfg.DATAROOT,
                           'annot',
                           'gt_{}.mat'.format(cfg.TEST_SET))
    gt_dict = loadmat(gt_file)
    dataset_joints = gt_dict['dataset_joints']
    jnt_missing = gt_dict['jnt_missing']
    pos_gt_src = gt_dict['pos_gt_src']
    headboxes_src = gt_dict['headboxes_src']

    pos_pred_src = np.transpose(preds, [1, 2, 0])

    head = np.where(dataset_joints == 'head')[1][0]
    lsho = np.where(dataset_joints == 'lsho')[1][0]
    lelb = np.where(dataset_joints == 'lelb')[1][0]
    lwri = np.where(dataset_joints == 'lwri')[1][0]
    lhip = np.where(dataset_joints == 'lhip')[1][0]
    lkne = np.where(dataset_joints == 'lkne')[1][0]
    lank = np.where(dataset_joints == 'lank')[1][0]

    rsho = np.where(dataset_joints == 'rsho')[1][0]
    relb = np.where(dataset_joints == 'relb')[1][0]
    rwri = np.where(dataset_joints == 'rwri')[1][0]
    rkne = np.where(dataset_joints == 'rkne')[1][0]
    rank = np.where(dataset_joints == 'rank')[1][0]
    rhip = np.where(dataset_joints == 'rhip')[1][0]

    jnt_visible = 1 - jnt_missing
    uv_error = pos_pred_src - pos_gt_src
    uv_err = np.linalg.norm(uv_error, axis=1)
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= SC_BIAS
    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    jnt_count = np.sum(jnt_visible, axis=1)
    less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                      jnt_visible)
    PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

    # Save
    rng = np.arange(0, 0.5+0.01, 0.01)
    pckAll = np.zeros((len(rng), cfg.kp_dim))

    for r in range(len(rng)):
        threshold = rng[r]
        less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                          jnt_visible)
        pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                 jnt_count)

    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True

    jnt_count = np.ma.array(jnt_count, mask=False)
    jnt_count.mask[6:8] = True
    jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

    name_value = [
        ('Head', PCKh[head]),
        ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
        ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
        ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
        ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
        ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
        ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
        ('Mean', np.sum(PCKh * jnt_ratio)),
        ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
    ]
    name_value = OrderedDict(name_value)

    return name_value, name_value['Mean']

# TODO: coco_evaluate()

if __name__ == '__main__':
    args = parser.parse_args()
    valid(args)
