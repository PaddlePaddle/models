from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.98'

import paddle
import paddle.fluid as fluid
import numpy as np
import argparse
from reader import CityscapeDataset
import reader
import models
import sys


def add_argument(name, type, default, help):
    parser.add_argument('--' + name, default=default, type=type, help=help)


def add_arguments():
    add_argument('total_step', int, -1,
                 "Number of the step to be evaluated, -1 for full evaluation.")
    add_argument('init_weights_path', str, None,
                 "Path of the weights to evaluate.")
    add_argument('dataset_path', str, None, "Cityscape dataset path.")
    add_argument('verbose', bool, False, "Print mIoU for each step if verbose.")
    add_argument('use_gpu', bool, True, "Whether use GPU or CPU.")


def mean_iou(pred, label):
    label = fluid.layers.elementwise_min(
        label, fluid.layers.assign(np.array(
            [num_classes], dtype=np.int32)))
    label_ignore = (label == num_classes).astype('int32')
    label_nignore = (label != num_classes).astype('int32')

    pred = pred * label_nignore + label_ignore * num_classes

    miou, wrong, correct = fluid.layers.mean_iou(pred, label, num_classes + 1)
    return miou, wrong, correct


def load_model():
    if args.init_weights_path.endswith('/'):
        fluid.io.load_params(
            exe, dirname=args.init_weights_path, main_program=tp)
    else:
        fluid.io.load_params(
            exe, dirname="", filename=args.init_weights_path, main_program=tp)


CityscapeDataset = reader.CityscapeDataset

parser = argparse.ArgumentParser()
add_arguments()

args = parser.parse_args()

models.clean()
models.is_train = False
deeplabv3p = models.deeplabv3p

image_shape = [1025, 2049]
eval_shape = [1024, 2048]

sp = fluid.Program()
tp = fluid.Program()
batch_size = 1
reader.default_config['crop_size'] = -1
reader.default_config['shuffle'] = False
num_classes = 19

with fluid.program_guard(tp, sp):
    img = fluid.layers.data(name='img', shape=[3, 0, 0], dtype='float32')
    label = fluid.layers.data(name='label', shape=eval_shape, dtype='int32')
    img = fluid.layers.resize_bilinear(img, image_shape)
    logit = deeplabv3p(img)
    logit = fluid.layers.resize_bilinear(logit, eval_shape)
    pred = fluid.layers.argmax(logit, axis=1).astype('int32')
    miou, out_wrong, out_correct = mean_iou(pred, label)

tp = tp.clone(True)
fluid.memory_optimize(
    tp,
    print_log=False,
    skip_opt_set=[pred.name, miou, out_wrong, out_correct],
    level=1)

place = fluid.CPUPlace()
if args.use_gpu:
    place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(sp)

if args.init_weights_path:
    print("load from:", args.init_weights_path)
    load_model()

dataset = CityscapeDataset(args.dataset_path, 'val')
if args.total_step == -1:
    total_step = len(dataset.label_files)
else:
    total_step = args.total_step

batches = dataset.get_batch_generator(batch_size, total_step)

sum_iou = 0
all_correct = np.array([0], dtype=np.int64)
all_wrong = np.array([0], dtype=np.int64)

for i, imgs, labels, names in batches:
    result = exe.run(tp,
                     feed={'img': imgs,
                           'label': labels},
                     fetch_list=[pred, miou, out_wrong, out_correct])
    wrong = result[2][:-1] + all_wrong
    right = result[3][:-1] + all_correct
    all_wrong = wrong.copy()
    all_correct = right.copy()
    mp = (wrong + right) != 0
    miou2 = np.mean((right[mp] * 1.0 / (right[mp] + wrong[mp])))
    if args.verbose:
        print('step: %s, mIoU: %s' % (i + 1, miou2))
    else:
        print('\rstep: %s, mIoU: %s' % (i + 1, miou2))
        sys.stdout.flush()
