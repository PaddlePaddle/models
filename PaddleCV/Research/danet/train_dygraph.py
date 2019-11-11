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

    pred = fluid.layers.softmax(pred, use_cudnn=False)
    loss1 = fluid.layers.cross_entropy(pred, label, ignore_index=255)

    pred2 = fluid.layers.softmax(pred2, use_cudnn=False)
    loss2 = fluid.layers.cross_entropy(pred2, label, ignore_index=255)

    pred3 = fluid.layers.softmax(pred3, use_cudnn=False)
    loss3 = fluid.layers.cross_entropy(pred3, label, ignore_index=255)

    label.stop_gradient = True
    return loss1 + loss2 + loss3


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

    batch_size = args.batch_size
    num_epochs = args.epoch_num
    num_classes = args.num_classes
    data_root = args.data_folder
    num = fluid.core.get_cuda_device_count()
    print('GPU设备数量： {}'.format(num))

    # program
    start_prog = fluid.default_startup_program()
    train_prog = fluid.default_main_program()

    start_prog.random_seed = args.seed
    train_prog.random_seed = args.seed

    logging.basicConfig(level=logging.INFO,
                        filename='DANet_{}_train.log'.format(args.backbone),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info('DANet')
    logging.info(args)

    place = fluid.CUDAPlace(0) if args.cuda else fluid.CPUPlace()
    train_loss_title = 'Train_loss'
    test_loss_title = 'Test_loss'

    train_iou_title = 'Train_mIOU'
    test_iou_title = 'Test_mIOU'

    plot_loss = Ploter(train_loss_title, test_loss_title)
    plot_iou = Ploter(train_iou_title, test_iou_title)

    with fluid.dygraph.guard(place):

        model = get_model(args)
        x = np.random.randn(batch_size, 3, 224, 224).astype('float32')
        x = fluid.dygraph.to_variable(x)
        model(x)

        # 加载预训练模型
        if args.load_pretrained_model:
            save_dir = 'checkpoint/DANet101_pretrained_model_paddle1.6'
            if os.path.exists(save_dir + '.pdparams'):
                param, _ = fluid.load_dygraph(save_dir)
                model.set_dict(param)
                assert len(param) == len(model.state_dict()), "参数量不一致，加载参数失败，" \
                                                              "请核对模型是否初始化/模型是否一致"
                print('load pretrained model!')

        # 加载最优模型
        if args.load_better_model:
            save_dir = 'checkpoint/DANet101_better_model_paddle1.6'
            if os.path.exists(save_dir + '.pdparams'):
                param, _ = fluid.load_dygraph(save_dir)
                model.set_dict(param)
                assert len(param) == len(model.state_dict()), "参数量不一致，加载参数失败，" \
                                                              "请核对模型是否初始化/模型是否一致"
                print('load better model!')

        optimizer = optimizer_setting(args)
        train_data = cityscapes_train(data_root=data_root,
                                      base_size=args.base_size,
                                      crop_size=args.crop_size,
                                      scale=args.scale,
                                      xmap=True,
                                      batch_size=batch_size,
                                      gpu_num=num)
        batch_train_data = paddle.batch(paddle.reader.shuffle(
            train_data, buf_size=batch_size * 64),
            batch_size=batch_size,
            drop_last=True)

        val_data = cityscapes_val(data_root=data_root,
                                  base_size=args.base_size,
                                  crop_size=args.crop_size,
                                  scale=args.scale,
                                  xmap=True)
        batch_test_data = paddle.batch(val_data,
                                       batch_size=batch_size,
                                       drop_last=True)

        train_iou_manager = fluid.metrics.Accuracy()
        train_avg_loss_manager = fluid.metrics.Accuracy()
        test_iou_manager = fluid.metrics.Accuracy()
        test_avg_loss_manager = fluid.metrics.Accuracy()

        better_miou_train = 0
        better_miou_test = 0

        for epoch in range(num_epochs):
            prev_time = datetime.now()
            train_avg_loss_manager.reset()
            train_iou_manager.reset()
            for batch_id, data in enumerate(batch_train_data()):
                image = np.array([x[0] for x in data]).astype('float32')
                label = np.array([x[1] for x in data]).astype('int64')

                image = fluid.dygraph.to_variable(image)
                label = fluid.dygraph.to_variable(label)
                label.stop_gradient = True
                pred, pred2, pred3 = model(image)
                train_loss = loss_fn(pred, pred2, pred3, label, num_classes=num_classes)
                train_avg_loss = fluid.layers.mean(train_loss)
                miou, wrong, correct = mean_iou(pred, label, num_classes=num_classes)
                train_avg_loss.backward()
                optimizer.minimize(train_avg_loss)
                model.clear_gradients()
                train_iou_manager.update(miou.numpy(), weight=batch_size*num)
                train_avg_loss_manager.update(train_avg_loss.numpy(), weight=batch_size*num)
                batch_train_str = "epoch: {}, batch: {}, train_avg_loss: {:.6f}, " \
                                  "train_miou: {:.6f}.".format(epoch + 1,
                                                               batch_id + 1,
                                                               train_avg_loss.numpy()[0],
                                                               miou.numpy()[0])
                if batch_id % 100 == 0:
                    logging.info(batch_train_str)
                    print(batch_train_str)
            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = " Time %02d:%02d:%02d" % (h, m, s)
            train_str = "\nepoch: {}, train_avg_loss: {:.6f}, " \
                        "train_miou: {:.6f}.".format(epoch + 1,
                                                     train_avg_loss_manager.eval()[0],
                                                     train_iou_manager.eval()[0])
            print(train_str + time_str + '\n')
            logging.info(train_str + time_str + '\n')
            plot_loss.append(train_loss_title, epoch, train_avg_loss_manager.eval()[0])
            plot_loss.plot('./DANet_loss.jpg')
            plot_iou.append(train_iou_title, epoch, train_iou_manager.eval()[0])
            plot_iou.plot('./DANet_miou.jpg')
            fluid.dygraph.save_dygraph(model.state_dict(), 'checkpoint/DANet_epoch_new')
            # save_model
            if better_miou_train < train_iou_manager.eval()[0]:
                shutil.rmtree('checkpoint/DAnet_better_train_{:.4f}.pdparams'.format(better_miou_train), ignore_errors=True)
                better_miou_train = train_iou_manager.eval()[0]
                fluid.dygraph.save_dygraph(model.state_dict(),
                                           'checkpoint/DAnet_better_train_{:.4f}'.format(better_miou_train))

            ########## test ############
            model.eval()
            test_iou_manager.reset()
            test_avg_loss_manager.reset()
            prev_time = datetime.now()
            for (batch_id, data) in enumerate(batch_test_data()):
                image = np.array([x[0] for x in data]).astype('float32')
                label = np.array([x[1] for x in data]).astype('int64')

                image = fluid.dygraph.to_variable(image)
                label = fluid.dygraph.to_variable(label)

                label.stop_gradient = True
                pred, pred2, pred3 = model(image)
                test_loss = loss_fn(pred, pred2, pred3, label, num_classes=num_classes)
                test_avg_loss = fluid.layers.mean(test_loss)
                miou, wrong, correct = mean_iou(pred, label, num_classes=num_classes)
                test_iou_manager.update(miou.numpy(), weight=batch_size*num)
                test_avg_loss_manager.update(test_avg_loss.numpy(), weight=batch_size*num)
                batch_test_str = "epoch: {}, batch: {}, test_avg_loss: {:.6f}, " \
                                 "test_miou: {:.6f}.".format(epoch + 1, batch_id + 1,
                                                             test_avg_loss.numpy()[0],
                                                             miou.numpy()[0])
                if batch_id % 20 == 0:
                    logging.info(batch_test_str)
                    print(batch_test_str)
            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = " Time %02d:%02d:%02d" % (h, m, s)
            test_str = "\nepoch: {}, test_avg_loss: {:.6f}, " \
                       "test_miou: {:.6f}.".format(epoch + 1,
                                                   test_avg_loss_manager.eval()[0],
                                                   test_iou_manager.eval()[0])
            print(test_str + time_str + '\n')
            logging.info(test_str + time_str + '\n')
            plot_loss.append(test_loss_title, epoch, test_avg_loss_manager.eval()[0])
            plot_loss.plot('./DANet_loss.jpg')
            plot_iou.append(test_iou_title, epoch, test_iou_manager.eval()[0])
            plot_iou.plot('./DANet_miou.jpg')
            model.train()
            # save_model
            if better_miou_test < test_iou_manager.eval()[0]:
                shutil.rmtree('checkpoint/DAnet_better_test_{:.4f}.pdparams'.format(better_miou_test), ignore_errors=True)
                better_miou_test = test_iou_manager.eval()[0]
                fluid.dygraph.save_dygraph(model.state_dict(),
                                           'checkpoint/DAnet_better_test_{:.4f}'.format(better_miou_test))


if __name__ == '__main__':
    options = Options()
    args = options.parse()
    options.print_args()
    main(args)
