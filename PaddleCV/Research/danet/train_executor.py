# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


import os

os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.99"

import paddle.fluid as fluid
import numpy as np
import paddle
import logging
import shutil
from datetime import datetime
from paddle.utils import Ploter

from danet import DANet
from options import Options
from utils.cityscapes_data import cityscapes_train
from utils.cityscapes_data import cityscapes_val
from utils.lr_scheduler import Lr


def get_model(args):
    model = DANet('DANet',
                  backbone=args.backbone,
                  num_classes=args.num_classes,
                  batch_size=args.batch_size,
                  dilated=args.dilated,
                  multi_grid=args.multi_grid,
                  multi_dilation=args.multi_dilation)
    return model


def mean_iou(pred, label, num_classes=19):
    label = fluid.layers.elementwise_min(fluid.layers.cast(label, np.int32),
                                         fluid.layers.assign(np.array([num_classes], dtype=np.int32)))
    label_ig = (label == num_classes).astype('int32')
    label_ng = (label != num_classes).astype('int32')
    pred = fluid.layers.cast(fluid.layers.argmax(pred, axis=1), 'int32')
    pred = pred * label_ng + label_ig * num_classes
    miou, wrong, correct = fluid.layers.mean_iou(pred, label, num_classes + 1)
    label.stop_gradient = True
    return miou, wrong, correct


def loss_fn(pred, pred2, pred3, label, num_classes=19):
    pred = fluid.layers.transpose(pred, perm=[0, 2, 3, 1])
    pred = fluid.layers.reshape(pred, [-1, num_classes])

    pred2 = fluid.layers.transpose(pred2, perm=[0, 2, 3, 1])
    pred2 = fluid.layers.reshape(pred2, [-1, num_classes])

    pred3 = fluid.layers.transpose(pred3, perm=[0, 2, 3, 1])
    pred3 = fluid.layers.reshape(pred3, [-1, num_classes])

    label = fluid.layers.reshape(label, [-1, 1])

    # loss1 = fluid.layers.softmax_with_cross_entropy(pred, label, ignore_index=255)
    # 以上方式会出现loss为NaN的情况
    pred = fluid.layers.softmax(pred, use_cudnn=False)
    loss1 = fluid.layers.cross_entropy(pred, label, ignore_index=255)

    pred2 = fluid.layers.softmax(pred2, use_cudnn=False)
    loss2 = fluid.layers.cross_entropy(pred2, label, ignore_index=255)

    pred3 = fluid.layers.softmax(pred3, use_cudnn=False)
    loss3 = fluid.layers.cross_entropy(pred3, label, ignore_index=255)

    label.stop_gradient = True
    return loss1 + loss2 + loss3


def save_model(save_dir, exe, program=None):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir, ignore_errors=True)
        os.makedirs(save_dir)
        fluid.io.save_persistables(exe, save_dir, program)
        print('已保存: {}'.format(os.path.basename(save_dir)))
    else:
        os.makedirs(save_dir)
        fluid.io.save_persistables(exe, save_dir, program)
        print('不存在，创建: {}'.format(os.path.basename(save_dir)))


def load_model(save_dir, exe, program=None):
    if os.path.exists(save_dir):
        fluid.io.load_persistables(exe, save_dir, program)
        print('存在, 加载成功')
    else:
        raise Exception('请核对地址')


def optimizer_setting(args):
    if args.weight_decay is not None:
        regular = fluid.regularizer.L2Decay(regularization_coeff=args.weight_decay)
    else:
        regular = None
    if args.lr_scheduler == 'poly':
        lr_scheduler = Lr(lr_policy='poly',
                          base_lr=args.lr,
                          epoch_nums=args.epoch_num,
                          step_per_epoch=args.step_per_epoch,
                          power=args.lr_pow,
                          warm_up=args.warm_up,
                          warmup_epoch=args.warmup_epoch)
        decayed_lr = lr_scheduler.get_lr()
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = Lr(lr_policy='cosine',
                          base_lr=args.lr,
                          epoch_nums=args.epoch_num,
                          step_per_epoch=args.step_per_epoch,
                          warm_up=args.warm_up,
                          warmup_epoch=args.warmup_epoch)
        decayed_lr = lr_scheduler.get_lr()
    elif args.lr_scheduler == 'piecewise':
        lr_scheduler = Lr(lr_policy='piecewise',
                          base_lr=args.lr,
                          epoch_nums=args.epoch_num,
                          step_per_epoch=args.step_per_epoch,
                          warm_up=args.warm_up,
                          warmup_epoch=args.warmup_epoch,
                          decay_epoch=[50, 100, 150],
                          gamma=0.1)
        decayed_lr = lr_scheduler.get_lr()
    else:
        decayed_lr = args.lr
    return fluid.optimizer.MomentumOptimizer(learning_rate=decayed_lr,
                                             momentum=args.momentum,
                                             regularization=regular)


def main(args):
    image_shape = args.crop_size
    image = fluid.layers.data(name='image', shape=[3, image_shape, image_shape], dtype='float32')
    label = fluid.layers.data(name='label', shape=[image_shape, image_shape], dtype='int64')

    batch_size = args.batch_size
    epoch_num = args.epoch_num
    num_classes = args.num_classes
    data_root = args.data_folder
    num = fluid.core.get_cuda_device_count()
    print('GPU设备数量： {}'.format(num))

    # program
    start_prog = fluid.default_startup_program()
    train_prog = fluid.default_main_program()

    start_prog.random_seed = args.seed
    train_prog.random_seed = args.seed

    # clone 
    test_prog = train_prog.clone(for_test=True)

    logging.basicConfig(level=logging.INFO,
                        filename='DANet_{}_train.log'.format(args.backbone),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info('DANet')
    logging.info(args)

    with fluid.program_guard(train_prog, start_prog):
        with fluid.unique_name.guard():
            train_py_reader = fluid.io.PyReader(feed_list=[image, label],
                                                capacity=64,
                                                use_double_buffer=True,
                                                iterable=False)
            train_data = cityscapes_train(data_root=data_root,
                                          base_size=args.base_size,
                                          crop_size=args.crop_size,
                                          scale=args.scale,
                                          xmap=True,
                                          batch_size=batch_size,
                                          gpu_num=num)
            batch_train_data = paddle.batch(paddle.reader.shuffle(
                train_data, buf_size=batch_size * 16),
                batch_size=batch_size,
                drop_last=True)
            train_py_reader.decorate_sample_list_generator(batch_train_data)

            model = get_model(args)
            pred, pred2, pred3 = model(image)
            train_loss = loss_fn(pred, pred2, pred3, label, num_classes=num_classes)
            train_avg_loss = fluid.layers.mean(train_loss)
            optimizer = optimizer_setting(args)
            optimizer.minimize(train_avg_loss)
            # miou不是真实的
            miou, wrong, correct = mean_iou(pred, label, num_classes=num_classes)

    with fluid.program_guard(test_prog, start_prog):
        with fluid.unique_name.guard():
            test_py_reader = fluid.io.PyReader(feed_list=[image, label],
                                               capacity=64,
                                               iterable=False,
                                               use_double_buffer=True)
            val_data = cityscapes_val(data_root=data_root,
                                      base_size=args.base_size,
                                      crop_size=args.crop_size,
                                      scale=args.scale,
                                      xmap=True)
            batch_test_data = paddle.batch(val_data,
                                           batch_size=batch_size,
                                           drop_last=True)
            test_py_reader.decorate_sample_list_generator(batch_test_data)

            model = get_model(args)
            pred, pred2, pred3 = model(image)
            test_loss = loss_fn(pred, pred2, pred3, label, num_classes=num_classes)
            test_avg_loss = fluid.layers.mean(test_loss)
            # miou不是真实的
            miou, wrong, correct = mean_iou(pred, label, num_classes=num_classes)

    place = fluid.CUDAPlace(0) if args.cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(start_prog)

    if args.use_data_parallel:
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = fluid.core.get_cuda_device_count()
        exec_strategy.num_iteration_per_drop_scope = 100
        build_strategy = fluid.BuildStrategy()
        build_strategy.sync_batch_norm = True
        print("sync_batch_norm = True!")
        compiled_train_prog = fluid.compiler.CompiledProgram(train_prog).with_data_parallel(
            loss_name=train_avg_loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
    else:
        compiled_train_prog = fluid.compiler.CompiledProgram(train_prog)

    # 加载预训练模型
    if args.load_pretrained_model:
        save_dir = 'checkpoint/DANet101_better_model_paddle1.5.2'
        if os.path.exists(save_dir):
            load_model(save_dir, exe, program=train_prog)
            print('load pretrained model!')

    # 加载最优模型
    if args.load_better_model:
        save_dir = 'checkpoint/DANet101_better_model_paddle1.5.2'
        if os.path.exists(save_dir):
            load_model(save_dir, exe, program=train_prog)
            print('load better model!')

    train_iou_manager = fluid.metrics.Accuracy()
    train_avg_loss_manager = fluid.metrics.Accuracy()
    test_iou_manager = fluid.metrics.Accuracy()
    test_avg_loss_manager = fluid.metrics.Accuracy()
    better_miou_train = 0
    better_miou_test = 0

    train_loss_title = 'Train_loss'
    test_loss_title = 'Test_loss'

    train_iou_title = 'Train_mIOU'
    test_iou_title = 'Test_mIOU'

    plot_loss = Ploter(train_loss_title, test_loss_title)
    plot_iou = Ploter(train_iou_title, test_iou_title)

    for epoch in range(epoch_num):
        prev_time = datetime.now()
        train_avg_loss_manager.reset()
        train_iou_manager.reset()
        logging.info('training, epoch = {}'.format(epoch + 1))
        train_py_reader.start()
        batch_id = 0
        while True:
            try:
                train_fetch_list = [train_avg_loss, miou, wrong, correct]
                train_avg_loss_value, train_iou_value, w, c = exe.run(
                    program=compiled_train_prog,
                    fetch_list=train_fetch_list)

                train_iou_manager.update(train_iou_value, weight=batch_size * num)
                train_avg_loss_manager.update(train_avg_loss_value, weight=batch_size * num)
                batch_train_str = "epoch: {}, batch: {}, train_avg_loss: {:.6f}, " \
                                  "train_miou: {:.6f}.".format(epoch + 1,
                                                                 batch_id + 1,
                                                                 train_avg_loss_value[0],
                                                                 train_iou_value[0])
                save_dir = './checkpoint/DAnet_better_train_{:.4f}'.format(22.5)
                save_model(save_dir, exe, program=train_prog)
                if batch_id % 40 == 0:
                    logging.info(batch_train_str)
                    print(batch_train_str)
                batch_id += 1
            except fluid.core.EOFException:
                train_py_reader.reset()
                break
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = " Time %02d:%02d:%02d" % (h, m, s)
        train_str = "epoch: {}, train_avg_loss: {:.6f}, " \
                    "train_miou: {:.6f}.".format(epoch + 1,
                                                 train_avg_loss_manager.eval()[0],
                                                 train_iou_manager.eval()[0])
        print(train_str + time_str + '\n')
        logging.info(train_str + time_str)
        plot_loss.append(train_loss_title, epoch, train_avg_loss_manager.eval()[0])
        plot_loss.plot('./DANet_loss.jpg')
        plot_iou.append(train_iou_title, epoch, train_iou_manager.eval()[0])
        plot_iou.plot('./DANet_miou.jpg')

        # save_model
        if better_miou_train < train_iou_manager.eval()[0]:
            shutil.rmtree('./checkpoint/DAnet_better_train_{:.4f}'.format(better_miou_train),
                          ignore_errors=True)
            better_miou_train = train_iou_manager.eval()[0]
            logging.warning(
                '-----------train---------------better_train: {:.6f}, epoch: {}, -----------successful save train model!\n'.format(
                    better_miou_train, epoch + 1))
            save_dir = './checkpoint/DAnet_better_train_{:.4f}'.format(better_miou_train)
            save_model(save_dir, exe, program=train_prog)
        if (epoch + 1) % 5 == 0:
            save_dir = './checkpoint/DAnet_epoch_train'
            save_model(save_dir, exe, program=train_prog)

        # test
        test_py_reader.start()
        test_iou_manager.reset()
        test_avg_loss_manager.reset()
        prev_time = datetime.now()
        logging.info('testing, epoch = {}'.format(epoch + 1))
        batch_id = 0
        while True:
            try:
                test_fetch_list = [test_avg_loss, miou, wrong, correct]
                test_avg_loss_value, test_iou_value, _, _ = exe.run(program=test_prog,
                                                                                                fetch_list=test_fetch_list)
                test_iou_manager.update(test_iou_value, weight=batch_size * num)
                test_avg_loss_manager.update(test_avg_loss_value, weight=batch_size * num)
                batch_test_str = "epoch: {}, batch: {}, test_avg_loss: {:.6f}, " \
                                 "test_miou: {:.6f}. ".format(epoch + 1,
                                                              batch_id + 1,
                                                              test_avg_loss_value[0],
                                                              test_iou_value[0])
                if batch_id % 40 == 0:
                    logging.info(batch_test_str)
                    print(batch_test_str)
                batch_id += 1
            except fluid.core.EOFException:
                test_py_reader.reset()
                break
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = " Time %02d:%02d:%02d" % (h, m, s)
        test_str = "epoch: {}, test_avg_loss: {:.6f}, " \
                   "test_miou: {:.6f}.".format(epoch + 1,
                                               test_avg_loss_manager.eval()[0],
                                               test_iou_manager.eval()[0])
        print(test_str + time_str + '\n')
        logging.info(test_str + time_str)
        plot_loss.append(test_loss_title, epoch, test_avg_loss_manager.eval()[0])
        plot_loss.plot('./DANet_loss.jpg')
        plot_iou.append(test_iou_title, epoch, test_iou_manager.eval()[0])
        plot_iou.plot('./DANet_miou.jpg')

        # save_model_infer
        if better_miou_test < test_iou_manager.eval()[0]:
            shutil.rmtree('./checkpoint/infer/DAnet_better_test_{:.4f}'.format(better_miou_test),
                          ignore_errors=True)
            better_miou_test = test_iou_manager.eval()[0]
            logging.warning(
                '------------test-------------infer better_test: {:.6f}, epoch: {}, ----------------successful save infer model!\n'.format(
                    better_miou_test, epoch + 1))
            save_dir = './checkpoint/infer/DAnet_better_test_{:.4f}'.format(better_miou_test)
            # save_model(save_dir, exe, program=test_prog)
            fluid.io.save_inference_model(save_dir, [image.name], [pred, pred2, pred3], exe)
            print('successful save infer model!')


if __name__ == '__main__':
    options = Options()
    args = options.parse()
    options.print_args()
    main(args)
