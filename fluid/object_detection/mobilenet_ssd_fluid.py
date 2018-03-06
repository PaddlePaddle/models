import os

import paddle.v2 as paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
import reader
import numpy as np

parameter_attr = ParamAttr(initializer=MSRA())


def conv_bn_layer(input,
                  filter_size,
                  num_filters,
                  stride,
                  padding,
                  channels=None,
                  num_groups=1,
                  act='relu',
                  use_cudnn=True):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        groups=num_groups,
        act=None,
        use_cudnn=use_cudnn,
        param_attr=parameter_attr,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv, act=act)


def depthwise_separable(input, num_filters1, num_filters2, num_groups, stride,
                        scale):
    """
    """
    depthwise_conv = conv_bn_layer(
        input=input,
        filter_size=3,
        num_filters=int(num_filters1 * scale),
        stride=stride,
        padding=1,
        num_groups=int(num_groups * scale),
        use_cudnn=False)

    pointwise_conv = conv_bn_layer(
        input=depthwise_conv,
        filter_size=1,
        num_filters=int(num_filters2 * scale),
        stride=1,
        padding=0)
    return pointwise_conv


def extra_block(input, num_filters1, num_filters2, num_groups, stride, scale):
    """
    """
    pointwise_conv = conv_bn_layer(
        input=input,
        filter_size=1,
        num_filters=int(num_filters1 * scale),
        stride=1,
        num_groups=int(num_groups * scale),
        padding=0)

    normal_conv = conv_bn_layer(
        input=pointwise_conv,
        filter_size=3,
        num_filters=int(num_filters2 * scale),
        stride=2,
        num_groups=int(num_groups * scale),
        padding=1)
    return normal_conv


def mobile_net(img, img_shape, scale=1.0):

    # 300x300
    tmp = conv_bn_layer(
        img,
        filter_size=3,
        channels=3,
        num_filters=int(32 * scale),
        stride=2,
        padding=1)

    # 150x150
    tmp = depthwise_separable(
        tmp,
        num_filters1=32,
        num_filters2=64,
        num_groups=32,
        stride=1,
        scale=scale)

    tmp = depthwise_separable(
        tmp,
        num_filters1=64,
        num_filters2=128,
        num_groups=64,
        stride=2,
        scale=scale)

    # 75x75
    tmp = depthwise_separable(
        tmp,
        num_filters1=128,
        num_filters2=128,
        num_groups=128,
        stride=1,
        scale=scale)

    tmp = depthwise_separable(
        tmp,
        num_filters1=128,
        num_filters2=256,
        num_groups=128,
        stride=2,
        scale=scale)

    # 38x38
    tmp = depthwise_separable(
        tmp,
        num_filters1=256,
        num_filters2=256,
        num_groups=256,
        stride=1,
        scale=scale)

    tmp = depthwise_separable(
        tmp,
        num_filters1=256,
        num_filters2=512,
        num_groups=256,
        stride=2,
        scale=scale)

    # 19x19
    for i in range(5):
        tmp = depthwise_separable(
            tmp,
            num_filters1=512,
            num_filters2=512,
            num_groups=512,
            stride=1,
            scale=scale)
    module11 = tmp

    tmp = depthwise_separable(
        tmp,
        num_filters1=512,
        num_filters2=1024,
        num_groups=512,
        stride=2,
        scale=scale)

    # 10x10
    module13 = depthwise_separable(
        tmp,
        num_filters1=1024,
        num_filters2=1024,
        num_groups=1024,
        stride=1,
        scale=scale)

    module14 = extra_block(
        module13,
        num_filters1=256,
        num_filters2=512,
        num_groups=1,
        stride=2,
        scale=scale)

    # 5x5
    module15 = extra_block(
        module14,
        num_filters1=128,
        num_filters2=256,
        num_groups=1,
        stride=2,
        scale=scale)

    # 3x3
    module16 = extra_block(
        module15,
        num_filters1=128,
        num_filters2=256,
        num_groups=1,
        stride=2,
        scale=scale)

    # 2x2
    module17 = extra_block(
        module16,
        num_filters1=64,
        num_filters2=128,
        num_groups=1,
        stride=2,
        scale=scale)

    mbox_locs, mbox_confs, box, box_var = fluid.layers.multi_box_head(
        inputs=[module11, module13, module14, module15, module16, module17],
        image=img,
        num_classes=21,
        min_ratio=20,
        max_ratio=90,
        aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2., 3.], [2., 3.]],
        base_size=img_shape[2],
        offset=0.5,
        flip=True,
        clip=True)

    return mbox_locs, mbox_confs, box, box_var


def train(train_file_list,
          val_file_list,
          data_args,
          learning_rate,
          batch_size,
          num_passes,
          model_save_dir='model',
          init_model_path=None):
    image_shape = [3, data_args.resize_h, data_args.resize_w]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1], dtype='float32', lod_level=1)

    mbox_locs, mbox_confs, box, box_var = mobile_net(image, image_shape)
    nmsed_out = fluid.layers.detection_output(mbox_locs, mbox_confs, box,
                                              box_var)
    loss_vec = fluid.layers.ssd_loss(mbox_locs, mbox_confs, gt_box, gt_label,
                                     box, box_var)
    loss = fluid.layers.nn.reduce_sum(loss_vec)

    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.learning_rate_decay.exponential_decay(
            learning_rate=learning_rate,
            decay_steps=40000,
            decay_rate=0.1,
            staircase=True),
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(5 * 1e-5), )
    opts = optimizer.minimize(loss)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    train_reader = paddle.batch(
        reader.train(data_args, train_file_list), batch_size=batch_size)
    test_reader = paddle.batch(
        reader.test(data_args, train_file_list), batch_size=batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, gt_box, gt_label])

    for pass_id in range(num_passes):
        for batch_id, data in enumerate(train_reader()):
            loss_v = exe.run(fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[loss])
            if batch_id % 50 == 0:
                print("Pass {0}, batch {1}, loss {2}".format(pass_id, batch_id,
                                                             np.sum(loss_v)))
        if pass_id % 1 == 0:
            model_path = os.path.join(model_save_dir, str(pass_id))
            print 'save models to %s' % (model_path)
            fluid.io.save_inference_model(model_path, ['image'], [nmsed_out],
                                          exe)


if __name__ == '__main__':
    data_args = reader.Settings(
        data_dir='./data',
        label_file='label_list',
        resize_h=300,
        resize_w=300,
        mean_value=[104, 117, 124])
    train(
        train_file_list='./data/trainval.txt',
        val_file_list='./data/test.txt',
        data_args=data_args,
        learning_rate=0.001,
        batch_size=32,
        num_passes=300)
