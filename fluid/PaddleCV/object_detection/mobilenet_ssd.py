import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr


def conv_bn(input,
            filter_size,
            num_filters,
            stride,
            padding,
            channels=None,
            num_groups=1,
            act='relu',
            use_cudnn=True):
    parameter_attr = ParamAttr(learning_rate=0.1, initializer=MSRA())
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


def mobile_net(num_classes, img, img_shape, scale=1.0):
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
        num_classes=num_classes,
        min_ratio=20,
        max_ratio=90,
        min_sizes=[60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
        max_sizes=[[], 150.0, 195.0, 240.0, 285.0, 300.0],
        aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2., 3.], [2., 3.]],
        base_size=img_shape[2],
        offset=0.5,
        flip=True)

    return mbox_locs, mbox_confs, box, box_var
