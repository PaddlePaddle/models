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


def add_argument(name, type, default, help):
    parser.add_argument('--' + name, default=default, type=type, help=help)


def add_arguments():
    add_argument('batch_size', int, 2,
                 "The number of images in each batch during training.")
    add_argument('train_crop_size', int, 769,
                 "'Image crop size during training.")
    add_argument('base_lr', float, 0.0001,
                 "The base learning rate for model training.")
    add_argument('total_step', int, 90000, "Number of the training step.")
    add_argument('init_weights_path', str, None,
                 "Path of the initial weights in paddlepaddle format.")
    add_argument('save_weights_path', str, None,
                 "Path of the saved weights during training.")
    add_argument('dataset_path', str, None, "Cityscape dataset path.")
    add_argument('parallel', bool, False, "using ParallelExecutor.")
    add_argument('use_gpu', bool, True, "Whether use GPU or CPU.")


def load_model():
    if args.init_weights_path.endswith('/'):
        fluid.io.load_params(
            exe, dirname=args.init_weights_path, main_program=tp)
    else:
        fluid.io.load_params(
            exe, dirname="", filename=args.init_weights_path, main_program=tp)


def save_model():
    if args.save_weights_path.endswith('/'):
        fluid.io.save_params(
            exe, dirname=args.save_weights_path, main_program=tp)
    else:
        fluid.io.save_params(
            exe, dirname="", filename=args.save_weights_path, main_program=tp)


def loss(logit, label):
    label_nignore = (label < num_classes).astype('float32')
    label = fluid.layers.elementwise_min(
        label,
        fluid.layers.assign(np.array(
            [num_classes - 1], dtype=np.int32)))
    logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
    logit = fluid.layers.reshape(logit, [-1, num_classes])
    label = fluid.layers.reshape(label, [-1, 1])
    label = fluid.layers.cast(label, 'int64')
    label_nignore = fluid.layers.reshape(label_nignore, [-1, 1])
    loss = fluid.layers.softmax_with_cross_entropy(logit, label)
    loss = loss * label_nignore
    no_grad_set.add(label_nignore.name)
    no_grad_set.add(label.name)
    return loss, label_nignore


CityscapeDataset = reader.CityscapeDataset
parser = argparse.ArgumentParser()

add_arguments()

args = parser.parse_args()

models.clean()
models.bn_momentum = 0.9997
models.dropout_keep_prop = 0.9
deeplabv3p = models.deeplabv3p

sp = fluid.Program()
tp = fluid.Program()
crop_size = args.train_crop_size
batch_size = args.batch_size
image_shape = [crop_size, crop_size]
reader.default_config['crop_size'] = crop_size
reader.default_config['shuffle'] = True
num_classes = 19
weight_decay = 0.00004

base_lr = args.base_lr
total_step = args.total_step

no_grad_set = set()

with fluid.program_guard(tp, sp):
    img = fluid.layers.data(
        name='img', shape=[3] + image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=image_shape, dtype='int32')
    logit = deeplabv3p(img)
    pred = fluid.layers.argmax(logit, axis=1).astype('int32')
    loss, mask = loss(logit, label)
    lr = fluid.layers.polynomial_decay(
        base_lr, total_step, end_learning_rate=0, power=0.9)
    area = fluid.layers.elementwise_max(
        fluid.layers.reduce_mean(mask),
        fluid.layers.assign(np.array(
            [0.1], dtype=np.float32)))
    loss_mean = fluid.layers.reduce_mean(loss) / area

    opt = fluid.optimizer.Momentum(
        lr,
        momentum=0.9,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=weight_decay), )
    retv = opt.minimize(loss_mean, startup_program=sp, no_grad_set=no_grad_set)

fluid.memory_optimize(
    tp, print_log=False, skip_opt_set=[pred.name, loss_mean.name], level=1)

place = fluid.CPUPlace()
if args.use_gpu:
    place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(sp)

if args.init_weights_path:
    print("load from:", args.init_weights_path)
    load_model()

dataset = CityscapeDataset(args.dataset_path, 'train')

if args.parallel:
    exe_p = fluid.ParallelExecutor(
        use_cuda=True, loss_name=loss_mean.name, main_program=tp)

batches = dataset.get_batch_generator(batch_size, total_step)

for i, imgs, labels, names in batches:
    if args.parallel:
        retv = exe_p.run(fetch_list=[pred.name, loss_mean.name],
                         feed={'img': imgs,
                               'label': labels})
    else:
        retv = exe.run(tp,
                       feed={'img': imgs,
                             'label': labels},
                       fetch_list=[pred, loss_mean])
    if i % 100 == 0:
        print("Model is saved to", args.save_weights_path)
        save_model()
    print("step %s, loss: %s" % (i, np.mean(retv[1])))

print("Training done. Model is saved to", args.save_weights_path)
save_model()
