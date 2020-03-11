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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os


def set_paddle_flags(flags):
    for key, value in flags.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect. 
set_paddle_flags({
    'FLAGS_eager_delete_tensor_gb': 0,  # enable GC 
    # You can omit the following settings, because the default
    # value of FLAGS_memory_fraction_of_eager_deletion is 1,
    # and default value of FLAGS_fast_eager_deletion_mode is 1 
    'FLAGS_memory_fraction_of_eager_deletion': 1,
    'FLAGS_fast_eager_deletion_mode': 1,
    # Setting the default used gpu memory
    'FLAGS_fraction_of_gpu_memory_to_use': 0.98
})

import paddle
import paddle.fluid as fluid
from paddle.fluid import profiler
import numpy as np
import argparse
from reader import CityscapeDataset
import reader
import models
import time
import contextlib
import paddle.fluid.profiler as profiler
import utility

parser = argparse.ArgumentParser()
add_arg = lambda *args: utility.add_arguments(*args, argparser=parser)

# yapf: disable
add_arg('batch_size',           int,    4,      "The number of images in each batch during training.")
add_arg('train_crop_size',      int,    769,    "Image crop size during training.")
add_arg('base_lr',              float,  0.001,  "The base learning rate for model training.")
add_arg('total_step',           int,    500000, "Number of the training step.")
add_arg('init_weights_path',    str,    None,   "Path of the initial weights in paddlepaddle format.")
add_arg('save_weights_path',    str,    None,   "Path of the saved weights during training.")
add_arg('dataset_path',         str,    None,   "Cityscape dataset path.")
add_arg('parallel',             bool,   True,   "using ParallelExecutor.")
add_arg('use_gpu',              bool,   True,   "Whether use GPU or CPU.")
add_arg('num_classes',          int,    19,     "Number of classes.")
add_arg('load_logit_layer',     bool,   True,   "Load last logit fc layer or not. If you are training with different number of classes, you should set to False.")
add_arg('memory_optimize',      bool,   True,   "Using memory optimizer.")
add_arg('norm_type',            str,    'bn',   "Normalization type, should be 'bn' or 'gn'.")
add_arg('profile',              bool,    False, "Enable profiler.")
add_arg('use_py_reader',        bool,    True,  "Use py reader.")
add_arg('use_multiprocessing',  bool,    False, "Use multiprocessing.")
add_arg("num_workers",          int,     8,     "The number of python processes used to read and preprocess data.")
# NOTE: args for profiler, used for benchmark
add_arg("profiler_path",        str,     '/tmp/profile_file2',  "the profiler output file path. (used for benchmark)")
parser.add_argument(
    '--enable_ce',
    action='store_true',
    help='If set, run the task with continuous evaluation logs. Users can ignore this agument.')
#yapf: enable

@contextlib.contextmanager
def profile_context(profile=True):
    if profile:
        with profiler.profiler('All', 'total', args.profiler_path):
            yield
    else:
        yield

def load_model():
    if os.path.isdir(args.init_weights_path):
        load_vars = [
            x for x in tp.list_vars()
            if isinstance(x, fluid.framework.Parameter) and x.name.find('logit') ==
            -1
        ]
        if args.load_logit_layer:
            fluid.io.load_params(
                exe, dirname=args.init_weights_path, main_program=tp)
        else:
            fluid.io.load_vars(exe, dirname=args.init_weights_path, vars=load_vars)
    else:
        fluid.io.load_params(
            exe,
            dirname="",
            filename=args.init_weights_path,
            main_program=tp)



def save_model():
    assert not os.path.isfile(args.save_weights_path)
    fluid.io.save_params(
        exe, dirname=args.save_weights_path, main_program=tp)


def loss(logit, label):
    label_nignore = fluid.layers.less_than(
        label.astype('float32'),
        fluid.layers.assign(np.array([num_classes], 'float32')),
        force_cpu=False).astype('float32')
    logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
    logit = fluid.layers.reshape(logit, [-1, num_classes])
    label = fluid.layers.reshape(label, [-1, 1])
    label = fluid.layers.cast(label, 'int64')
    label_nignore = fluid.layers.reshape(label_nignore, [-1, 1])
    logit = fluid.layers.softmax(logit, use_cudnn=False)
    loss = fluid.layers.cross_entropy(logit, label, ignore_index=255)
    label_nignore.stop_gradient = True
    label.stop_gradient = True
    return loss, label_nignore


args = parser.parse_args()
utility.print_arguments(args)
utility.check_gpu(args.use_gpu)

models.clean()
models.bn_momentum = 0.9997
models.dropout_keep_prop = 0.9
models.label_number = args.num_classes
models.default_norm_type = args.norm_type
deeplabv3p = models.deeplabv3p

sp = fluid.Program()
tp = fluid.Program()

crop_size = args.train_crop_size
batch_size = args.batch_size
image_shape = [crop_size, crop_size]
reader.default_config['crop_size'] = crop_size
reader.default_config['shuffle'] = True
num_classes = args.num_classes
weight_decay = 0.00004

base_lr = args.base_lr
total_step = args.total_step

# only for ce
if args.enable_ce:
    SEED = 102
    sp.random_seed = SEED
    tp.random_seed = SEED
    reader.default_config['shuffle'] = False

with fluid.program_guard(tp, sp):
    if args.use_py_reader:
        batch_size_each = batch_size // utility.get_device_count()
        py_reader = fluid.layers.py_reader(capacity=64,
                                        shapes=[[batch_size_each, 3] + image_shape, [batch_size_each] + image_shape],
                                        dtypes=['float32', 'int32'])
        img, label = fluid.layers.read_file(py_reader)
    else:
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
    loss_mean.persistable = True

    opt = fluid.optimizer.Momentum(
        lr,
        momentum=0.9,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=weight_decay))
    optimize_ops, params_grads = opt.minimize(loss_mean, startup_program=sp)
    # ir memory optimizer has some issues, we need to seed grad persistable to
    # avoid this issue
    for p,g in params_grads: g.persistable = True


exec_strategy = fluid.ExecutionStrategy()
exec_strategy.num_threads = utility.get_device_count()
exec_strategy.num_iteration_per_drop_scope = 100
build_strategy = fluid.BuildStrategy()
if args.memory_optimize:
    build_strategy.fuse_relu_depthwise_conv = True
    build_strategy.enable_inplace = True

place = fluid.CPUPlace()
if args.use_gpu:
    place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(sp)

if args.init_weights_path:
    print("load from:", args.init_weights_path)
    load_model()

dataset = reader.CityscapeDataset(args.dataset_path, 'train')

if args.parallel:
    binary = fluid.compiler.CompiledProgram(tp).with_data_parallel(
        loss_name=loss_mean.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)
else:
    binary = fluid.compiler.CompiledProgram(tp)

if args.use_py_reader:
    assert(batch_size % utility.get_device_count() == 0)
    def data_gen():
        batches = dataset.get_batch_generator(
            batch_size // utility.get_device_count(),
            total_step * utility.get_device_count(),
            use_multiprocessing=args.use_multiprocessing, num_workers=args.num_workers)
        for b in batches:
            yield b[0], b[1]
    py_reader.decorate_tensor_provider(data_gen)
    py_reader.start()
else:
    batches = dataset.get_batch_generator(batch_size, total_step, use_multiprocessing=True, num_workers=args.num_workers)
total_time = 0.0
epoch_idx = 0
train_loss = 0

with profile_context(args.profile):
    for i in range(total_step):
        epoch_idx += 1
        begin_time = time.time()
        if not args.use_py_reader:
            imgs, labels, names = next(batches)
            train_loss, = exe.run(binary,
                             feed={'img': imgs,
                                   'label': labels}, fetch_list=[loss_mean])
        else:
            train_loss, = exe.run(binary, fetch_list=[loss_mean])
        train_loss = np.mean(train_loss)
        end_time = time.time()
        total_time += end_time - begin_time

        if i % 100 == 0:
            print("Model is saved to", args.save_weights_path)
            save_model()
        print("step {:d}, loss: {:.6f}, step_time_cost: {:.3f} s".format(
            i, train_loss, end_time - begin_time))

print("Training done. Model is saved to", args.save_weights_path)
save_model()

if args.enable_ce:
    gpu_num = utility.get_device_count()
    print("kpis\teach_pass_duration_card%s\t%s" %
          (gpu_num, total_time / epoch_idx))
    print("kpis\ttrain_loss_card%s\t%s" % (gpu_num, train_loss))
