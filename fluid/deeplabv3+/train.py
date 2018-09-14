import os
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.98'
#os.environ['GLOG_logtostderr']='1'
#os.environ['GLOG_v']='1000'

import paddle
import paddle.fluid as fluid
import numpy as np
import argparse

from utils import Cityscape_dataset
import utils
Cityscape_dataset = utils.Cityscape_dataset

parser = argparse.ArgumentParser()

def add_argument(name, type, default, help):
    parser.add_argument('--'+name, default=default, type=type, help=help)
add_argument('batch_size', int, 2, "The number of images in each batch during training.")
add_argument('train_crop_size', int, 769, "'Image crop size during training.")
add_argument('base_lr', float, 0.0001, "The base learning rate for model training.")
add_argument('total_step', int, 90000, "Number of the training step.")
add_argument('init_weights_path', str, None, "Path of the initial weights in paddlepaddle format.")
add_argument('save_weights_path', str, None, "Path of the saved weights during training.")
add_argument('dataset_path', str, None, "Cityscape dataset path.")
add_argument('parallel', bool, False, "using ParallelExecutor.")

args = parser.parse_args()

import models
models.clean()
models.bn_momentum = 0.9997
models.dropout_keep_prop = 0.9
deeplabv3p = models.deeplabv3p

sp = fluid.Program()
tp = fluid.Program()
crop_size = args.train_crop_size
batch_size = args.batch_size
image_shape = [crop_size, crop_size]
utils.default_config['crop_size'] = crop_size
utils.default_config['shuffle'] = False
num_classes = 19
weight_decay = 0.00004

base_lr = args.base_lr
total_step = args.total_step


no_grad_set = set()
def loss(logit, label):
    #return fluid.layers.reduce_mean(logit)
    label_nignore = (label < num_classes).astype('float32')
    label = fluid.layers.elementwise_min(label, fluid.layers.assign(np.array([num_classes-1], dtype=np.int32)))
    logit = fluid.layers.transpose(logit, [0,2,3,1])
    logit = fluid.layers.reshape(logit, [-1, num_classes])
    label = fluid.layers.reshape(label, [-1, 1])
    label = fluid.layers.cast(label, 'int64')
    label_nignore = fluid.layers.reshape(label_nignore, [-1, 1])
    loss = fluid.layers.softmax_with_cross_entropy(logit, label)
    loss = loss * label_nignore
    no_grad_set.add(label_nignore.name)
    no_grad_set.add(label.name)
    return loss, label_nignore

with fluid.program_guard(tp, sp):
    img = fluid.layers.data(name='img', shape=[3]+image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=image_shape, dtype='int32')
    logit = deeplabv3p(img)
    pred = fluid.layers.argmax(logit, axis=1).astype('int32')
    loss, mask = loss(logit, label)
    lr = fluid.layers.polynomial_decay(base_lr, total_step, end_learning_rate=0, power=0.9)
    area = fluid.layers.elementwise_max(
        fluid.layers.reduce_mean(mask),
        fluid.layers.assign(np.array([0.1], dtype=np.float32)) )
    loss_mean = fluid.layers.reduce_mean(loss) / area

    opt = fluid.optimizer.Momentum(lr, momentum=0.9,
            regularization=fluid.regularizer.L2DecayRegularizer(regularization_coeff=weight_decay), )
    retv = opt.minimize(loss_mean, startup_program=sp, no_grad_set=no_grad_set)

fluid.memory_optimize(tp, print_log=False, skip_opt_set=[
    pred.name, loss_mean.name], level=1)

exe = fluid.Executor(fluid.CUDAPlace(0))
exe.run(sp)
if args.init_weights_path:
    print "load from:", args.init_weights_path
    fluid.io.load_params(exe, dirname=args.init_weights_path, main_program=tp)

utils.default_config['shuffle'] = True
dataset = Cityscape_dataset('train', args.dataset_path)

if args.parallel:
    print "Using ParallelExecutor."
    exe_p = fluid.ParallelExecutor(
        use_cuda=True,
        loss_name=loss_mean.name, main_program=tp)

def get_batch(n=10):
    for i in range(n):
        imgs, labels, names = dataset.get_batch(batch_size)
        labels = labels.astype(np.int32)[:,:,:,0]
        imgs = imgs[:,:,:,::-1].transpose(0,3,1,2).astype(np.float32) / (255.0/2) - 1
        yield i, imgs, labels, names


def get_batch_only_one(n=10):
    imgs, labels, names = dataset.get_batch(batch_size)
    labels = labels.astype(np.int32)[:,:,:,0]
    imgs = imgs[:,:,:,::-1].transpose(0,3,1,2).astype(np.float32) / (255.0/2) - 1
    for i in range(n):
        yield i, imgs, labels, names

batches= get_batch(total_step)

try:
    from prefetch_generator import BackgroundGenerator
    batches = BackgroundGenerator(batches, 100)
except:
    print "You can install 'prefetch_generator' for acceleration of data reading."

for i, imgs, labels, names in batches:
    if args.parallel:
        retv = exe_p.run(fetch_list=[pred.name, loss_mean.name], feed={ 'img':imgs, 'label':labels})
    else:
        retv = exe.run(tp, feed={ 'img':imgs, 'label':labels}, fetch_list=[pred, loss_mean])
    if i % 100 == 0:
        print "Model is saved to", args.save_weights_path
        fluid.io.save_params(exe, dirname=args.save_weights_path, main_program=tp)
    print i, np.mean(retv[1])

print "Training done. Model is saved to", args.save_weights_path
fluid.io.save_params(exe, dirname=args.save_weights_path, main_program=tp)
