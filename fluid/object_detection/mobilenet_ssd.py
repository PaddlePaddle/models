import os
import paddle.v2 as paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
import reader
import numpy as np
import load_model as load_model

parameter_attr = ParamAttr(initializer=MSRA())


def conv_bn(input,
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
    depthwise_conv = conv_bn(
        input=input,
        filter_size=3,
        num_filters=int(num_filters1 * scale),
        stride=stride,
        padding=1,
        num_groups=int(num_groups * scale),
        use_cudnn=False)

    pointwise_conv = conv_bn(
        input=depthwise_conv,
        filter_size=1,
        num_filters=int(num_filters2 * scale),
        stride=1,
        padding=0)
    return pointwise_conv


def extra_block(input, num_filters1, num_filters2, num_groups, stride, scale):
    # 1x1 conv
    pointwise_conv = conv_bn(
        input=input,
        filter_size=1,
        num_filters=int(num_filters1 * scale),
        stride=1,
        num_groups=int(num_groups * scale),
        padding=0)

    # 3x3 conv
    normal_conv = conv_bn(
        input=pointwise_conv,
        filter_size=3,
        num_filters=int(num_filters2 * scale),
        stride=2,
        num_groups=int(num_groups * scale),
        padding=1)
    return normal_conv


def mobile_net(img, img_shape, scale=1.0):
    # 300x300
    tmp = conv_bn(img, 3, int(32 * scale), 2, 1, 3)
    # 150x150
    tmp = depthwise_separable(tmp, 32, 64, 32, 1, scale)
    tmp = depthwise_separable(tmp, 64, 128, 64, 2, scale)
    # 75x75
    tmp = depthwise_separable(tmp, 128, 128, 128, 1, scale)
    tmp = depthwise_separable(tmp, 128, 256, 128, 2, scale)
    # 38x38
    tmp = depthwise_separable(tmp, 256, 256, 256, 1, scale)
    tmp = depthwise_separable(tmp, 256, 512, 256, 2, scale)

    # 19x19
    for i in range(5):
        tmp = depthwise_separable(tmp, 512, 512, 512, 1, scale)
    module11 = tmp
    tmp = depthwise_separable(tmp, 512, 1024, 512, 2, scale)

    # 10x10
    module13 = depthwise_separable(tmp, 1024, 1024, 1024, 1, scale)
    module14 = extra_block(module13, 256, 512, 1, 2, scale)
    # 5x5
    module15 = extra_block(module14, 128, 256, 1, 2, scale)
    # 3x3
    module16 = extra_block(module15, 128, 256, 1, 2, scale)
    # 2x2
    module17 = extra_block(module16, 64, 128, 1, 2, scale)
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
        name='gt_label', shape=[1], dtype='int32', lod_level=1)
    difficult = fluid.layers.data(
        name='gt_difficult', shape=[1], dtype='int32', lod_level=1)

    mbox_locs, mbox_confs, box, box_var = mobile_net(image, image_shape)
    nmsed_out = fluid.layers.detection_output(mbox_locs, mbox_confs, box,
                                              box_var)
    loss_vec = fluid.layers.ssd_loss(mbox_locs, mbox_confs, gt_box, gt_label,
                                     box, box_var)
    loss = fluid.layers.nn.reduce_sum(loss_vec)

    map_eval = fluid.evaluator.DetectionMAP(
        nmsed_out,
        gt_label,
        gt_box,
        difficult,
        21,
        overlap_threshold=0.5,
        evaluate_difficult=False,
        ap_version='11point')

    test_program = fluid.default_main_program().clone(for_test=True)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=learning_rate,
            decay_steps=40000,
            decay_rate=0.1,
            staircase=True),
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(0.0005), )
    opts = optimizer.minimize(loss)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    load_model.load_and_set_vars(place)

    train_reader = paddle.batch(
        reader.train(data_args, train_file_list), batch_size=batch_size)
    test_reader = paddle.batch(
        reader.test(data_args, train_file_list), batch_size=batch_size)
    feeder = fluid.DataFeeder(
        place=place, feed_list=[image, gt_box, gt_label, difficult])

    #print fluid.default_main_program()
    map, accum_map = map_eval.get_map_var()
    for pass_id in range(num_passes):
        map_eval.reset(exe)
        for batch_id, data in enumerate(train_reader()):
            loss_v, map_v, accum_map_v = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[loss, map, accum_map])
            print(
                "Pass {0}, batch {1}, loss {2}, cur_map {3}, map {4}"
                .format(pass_id, batch_id, loss_v[0], map_v[0], accum_map_v[0]))

        map_eval.reset(exe)
        test_map = None
        for _, data in enumerate(test_reader()):
            test_map = exe.run(test_program,
                               feed=feeder.feed(data),
                               fetch_list=[accum_map])
        print("Test {0}, map {1}".format(pass_id, test_map[0]))

        if pass_id % 10 == 0:
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
        learning_rate=0.004,
        batch_size=32,
        num_passes=300)
