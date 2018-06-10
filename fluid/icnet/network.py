import paddle.fluid as fluid
import numpy as np
import sys


def conv(input, k_h, k_w, c_o, s_h, s_w, relu=True, padding="VALID", group=1, biased=None, name=None, print_pad=False):
    act = None
    tmp = input
    if relu:
        act="relu"
    if padding == "SAME": 
        padding_h = max(k_h - s_h, 0)
        padding_w = max(k_w - s_w, 0)
        padding_top = padding_h / 2
        padding_left = padding_w / 2
        padding_bottom = padding_h - padding_top
        padding_right = padding_w - padding_left
        padding =  [0,0,0,0, padding_top, padding_bottom, padding_left, padding_right]
        tmp = fluid.layers.pad(tmp, padding)
    tmp = fluid.layers.conv2d(tmp,
            num_filters=c_o,
            filter_size=[k_h, k_w],
            stride=[s_h, s_w],
            groups=group,
            act=act,
            bias_attr=biased,
            use_cudnn=False,
            name=name)
    return tmp

def atrous_conv(input, k_h, k_w, c_o, dilation, relu=True, padding="VALID", group=1, biased=None, name=None):
    act = None
    if relu:
        act="relu"
    tmp = input
    if padding == "SAME": 
        padding_h = max(k_h - s_h, 0)
        padding_w = max(k_w - s_w, 0)
        padding_top = padding_h / 2
        padding_left = padding_w / 2
        padding_bottom = padding_h - padding_top
        padding_right = padding_w - padding_left
        padding =  [0,0,0,0, padding_top, padding_bottom, padding_left, padding_right]
        tmp = fluid.layers.pad(tmp, padding)

    tmp = fluid.layers.conv2d(input,
            num_filters=c_o,
            filter_size=[k_h, k_w],
            dilation=dilation,
            groups=group,
            act=act,
            bias_attr=biased,
            use_cudnn=False,
            name=name)
    return tmp

def zero_padding(input, padding):
    return fluid.layers.pad(input, [0, 0, 0, 0, padding, padding, padding, padding])

def bn(input, relu=False, name=None, is_test=False):
    act=None
    if relu:
        act = 'relu'
    tmp = fluid.layers.batch_norm(input, act=act, momentum=0.95, epsilon=1e-5, name=name, is_test=is_test)
    return tmp

def avg_pool(input, k_h, k_w, s_h, s_w, name=None, padding=0):
    temp = fluid.layers.pool2d(input,
            pool_size=[k_h, k_w],
            pool_type="avg",
            pool_stride=[s_h, s_w],
            pool_padding=padding,
            name=name)
    return temp

def max_pool(input, k_h, k_w, s_h, s_w, name=None, padding=0):
    temp = fluid.layers.pool2d(input,
            pool_size=[k_h, k_w],
            pool_type="max",
            pool_stride=[s_h, s_w],
            pool_padding=padding,
            name=name)
    return temp

def interp(input, out_shape):
    out_shape = list(out_shape.astype("int32"))
    return fluid.layers.upsampling_bilinear2d(input, out_shape=out_shape)


def icnet(args, data, num_classes, input_shape, is_test=False):
    data_sub2 = interp(data, out_shape=input_shape * 0.5)
    conv1_1_3_3_s2 = conv(data_sub2, 3, 3, 32, 2, 2, biased=False, padding='SAME', relu=False, name="conv1_1_3_3_s2")
    
    conv1_1_3_3_s2_bn = bn(conv1_1_3_3_s2, relu=True, name="conv1_1_3_3_s2_bn", is_test=is_test)
    

    conv1_2_3_3 = conv(conv1_1_3_3_s2_bn, 3, 3, 32, 1, 1, biased=False, padding='SAME', relu=False, name="conv1_2_3_3")
    conv1_2_3_3_bn = bn(conv1_2_3_3, relu=True, name="conv1_2_3_3_bn", is_test=is_test)
    conv1_3_3_3 = conv(conv1_2_3_3_bn, 3, 3, 64, 1, 1, biased=False, padding='SAME', relu=False, name="conv1_3_3_3")
    conv1_3_3_3_bn = bn(conv1_3_3_3, relu=True, name="conv1_3_3_3_bn", is_test=is_test)
    pool1_3_3_s2 = max_pool(conv1_3_3_3_bn, 3, 3, 2, 2, padding=[1, 1])
    conv2_1_1_1_proj = conv(pool1_3_3_s2, 1, 1, 128, 1, 1, biased=False, relu=False, name="conv2_1_1_1_proj")
    conv2_1_1_1_proj_bn = bn(conv2_1_1_1_proj, relu=False, name="conv2_1_1_1_proj_bn", is_test=is_test)


    conv2_1_1_1_reduce = conv(pool1_3_3_s2, 1, 1, 32, 1, 1, biased=False, relu=False, name="conv2_1_1_1_reduce")
    conv2_1_1_1_reduce_bn = bn(conv2_1_1_1_reduce, relu=True, name="conv2_1_1_1_reduce_bn", is_test=is_test)
    conv2_1_3_3 = conv(conv2_1_1_1_reduce_bn, 3, 3, 32, 1, 1, biased=False, relu=False, padding='SAME', name="conv2_1_3_3")
    conv2_1_3_3_bn = bn(conv2_1_3_3, relu=True, name="conv2_1_3_3_bn", is_test=is_test)
    conv2_1_1_1_increase = conv(conv2_1_3_3_bn, 1, 1, 128, 1, 1, biased=False, relu=False, name="conv2_1_1_1_increase")
    conv2_1_1_1_increase_bn = bn(conv2_1_1_1_increase, relu=False, name="conv2_1_1_1_increase_bn", is_test=is_test)

    conv2_1 = conv2_1_1_1_proj_bn + conv2_1_1_1_increase_bn
    conv2_1_relu = fluid.layers.relu(conv2_1, name="conv2_1_relu")
    
    conv2_2_1_1_reduce = conv(conv2_1_relu, 1, 1, 32, 1, 1, biased=False, relu=False, name="conv2_2_1_1_reduce")
    conv2_2_1_1_reduce_bn = bn(conv2_2_1_1_reduce, relu=True, name="conv2_2_1_1_reduce_bn", is_test=is_test)
    conv2_2_1_1_reduce_bn_pad = zero_padding(conv2_2_1_1_reduce_bn, padding=1)
    conv2_2_3_3 = conv(conv2_2_1_1_reduce_bn_pad, 3, 3, 32, 1, 1, biased=False, relu=False, name="conv2_2_3_3")
    conv2_2_3_3_bn = bn(conv2_2_3_3, relu=True, name="conv2_2_3_3_bn", is_test=is_test)
    conv2_2_1_1_increase = conv(conv2_2_3_3_bn, 1, 1, 128, 1, 1, biased=False, relu=False, name="conv2_2_1_1_increase")
    conv2_2_1_1_increase_bn = bn(conv2_2_1_1_increase, relu=False, name="conv2_2_1_1_increase_bn", is_test=is_test)

    conv2_2 = conv2_1_relu + conv2_2_1_1_increase_bn
    conv2_2_relu = fluid.layers.relu(conv2_2, name="conv2_2_relu")
    conv2_3_1_1_reduce = conv(conv2_2_relu, 1, 1, 32, 1, 1, biased=False, relu=False, name="conv2_3_1_1_reduce")
    conv2_3_1_1_reduce_bn = bn(conv2_3_1_1_reduce, relu=True, name="conv2_3_1_1_reduce_bn", is_test=is_test)
    conv2_3_1_1_reduce_bn_pad = zero_padding(conv2_3_1_1_reduce_bn, padding=1)
    conv2_3_3_3 = conv(conv2_3_1_1_reduce_bn_pad, 3, 3, 32, 1, 1, biased=False, relu=False, name="conv2_3_3_3")
    conv2_3_3_3_bn = bn(conv2_3_3_3, relu=True, name="conv2_3_3_3_bn", is_test=is_test)
    conv2_3_1_1_increase = conv(conv2_3_3_3_bn, 1, 1, 128, 1, 1, biased=False, relu=False, name="conv2_3_1_1_increase")
    conv2_3_1_1_increase_bn = bn(conv2_3_1_1_increase, relu=False, name="conv2_3_1_1_increase_bn", is_test=is_test)

    conv2_3 = conv2_2_relu + conv2_3_1_1_increase_bn
    
    conv2_3_relu = fluid.layers.relu(conv2_3, name="conv2_3_relu")
    conv3_1_1_1_proj = conv(conv2_3_relu, 1, 1, 256, 2, 2, biased=False, relu=False, name="conv3_1_1_1_proj")
    conv3_1_1_1_proj_bn = bn(conv3_1_1_1_proj, relu=False, name="conv3_1_1_1_proj_bn", is_test=is_test)

    conv3_1_1_1_reduce = conv(conv2_3_relu, 1, 1, 64, 2, 2, biased=False, relu=False, name="conv3_1_1_1_reduce")
    conv3_1_1_1_reduce_bn = bn(conv3_1_1_1_reduce, relu=True, name="conv3_1_1_1_reduce_bn", is_test=is_test)
    conv3_1_1_1_reduce_bn_pad = zero_padding(conv3_1_1_1_reduce_bn, padding=1)
    conv3_1_3_3 = conv(conv3_1_1_1_reduce_bn_pad, 3, 3, 64, 1, 1, biased=False, relu=False,  name="conv3_1_3_3")
    conv3_1_3_3_bn = bn(conv3_1_3_3, relu=True, name="conv3_1_3_3_bn", is_test=is_test)
    conv3_1_1_1_increase = conv(conv3_1_3_3_bn, 1, 1, 256, 1, 1, biased=False, relu=False, name="conv3_1_1_1_increase")
    conv3_1_1_1_increase_bn = bn(conv3_1_1_1_increase, relu=False, name="conv3_1_1_1_increase_bn", is_test=is_test)

    conv3_1 = conv3_1_1_1_proj_bn + conv3_1_1_1_increase_bn
    conv3_1_relu = fluid.layers.relu(conv3_1, name="conv3_1_relu")

    conv3_1_sub4 = interp(conv3_1_relu, out_shape=np.ceil(input_shape / 32))
    conv3_2_1_1_reduce = conv(conv3_1_sub4, 1, 1, 64, 1, 1, biased=False, relu=False, name="conv3_2_1_1_reduce")
    conv3_2_1_1_reduce_bn = bn(conv3_2_1_1_reduce, relu=True, name="conv3_2_1_1_reduce_bn", is_test=is_test)
    conv3_2_1_1_reduce_bn_pad = zero_padding(conv3_2_1_1_reduce_bn, padding=1)
    conv3_2_3_3 = conv(conv3_2_1_1_reduce_bn_pad, 3, 3, 64, 1, 1, biased=False, relu=False, name="conv3_2_3_3")
    conv3_2_3_3_bn = bn(conv3_2_3_3, relu=True, name="conv3_2_3_3_bn", is_test=is_test)
    conv3_2_1_1_increase = conv(conv3_2_3_3_bn, 1, 1, 256, 1, 1, biased=False, relu=False, name="conv3_2_1_1_increase")
    conv3_2_1_1_increase_bn = bn(conv3_2_1_1_increase, relu=False, name="conv3_2_1_1_increase_bn", is_test=is_test)

    conv3_2 = conv3_1_sub4 + conv3_2_1_1_increase_bn
    conv3_2_relu = fluid.layers.relu(conv3_2, name="conv3_2_relu")
    conv3_3_1_1_reduce = conv(conv3_2_relu, 1, 1, 64, 1, 1, biased=False, relu=False, name="conv3_3_1_1_reduce")
    conv3_3_1_1_reduce_bn = bn(conv3_3_1_1_reduce, relu=True, name="conv3_3_1_1_reduce_bn", is_test=is_test)
    conv3_3_1_1_reduce_bn_pad = zero_padding(conv3_3_1_1_reduce_bn, padding=1)
    conv3_3_3_3 = conv(conv3_3_1_1_reduce_bn_pad, 3, 3, 64, 1, 1, biased=False, relu=False, name="conv3_3_3_3")
    conv3_3_3_3_bn = bn(conv3_3_3_3, relu=True, name="conv3_3_3_3_bn", is_test=is_test)
    conv3_3_1_1_increase = conv(conv3_3_3_3_bn, 1, 1, 256, 1, 1, biased=False, relu=False, name="conv3_3_1_1_increase")
    conv3_3_1_1_increase_bn = bn(conv3_3_1_1_increase, relu=False, name="conv3_3_1_1_increase_bn", is_test=is_test)


    conv3_3 = conv3_2_relu + conv3_3_1_1_increase_bn
    conv3_3_relu = fluid.layers.relu(conv3_3, name="conv3_3_relu")
    conv3_4_1_1_reduce = conv(conv3_3_relu, 1, 1, 64, 1, 1, biased=False, relu=False, name="conv3_4_1_1_reduce")
    conv3_4_1_1_reduce_bn = bn(conv3_4_1_1_reduce, relu=True, name="conv3_4_1_1_reduce_bn", is_test=is_test)
    conv3_4_1_1_reduce_bn_pad = zero_padding(conv3_4_1_1_reduce_bn, padding=1)
    conv3_4_3_3 = conv(conv3_4_1_1_reduce_bn_pad, 3, 3, 64, 1, 1, biased=False, relu=False, name="conv3_4_3_3")
    conv3_4_3_3_bn = bn(conv3_4_3_3, relu=True, name="conv3_4_3_3_bn", is_test=is_test)
    conv3_4_1_1_increase = conv(conv3_4_3_3_bn, 1, 1, 256, 1, 1, biased=False, relu=False, name="conv3_4_1_1_increase")
    conv3_4_1_1_increase_bn = bn(conv3_4_1_1_increase, relu=False, name="conv3_4_1_1_increase_bn", is_test=is_test)

    conv3_4 = conv3_3_relu + conv3_4_1_1_increase_bn
    conv3_4_relu = fluid.layers.relu(conv3_4, name="conv3_4_relu")
    conv4_1_1_1_proj = conv(conv3_4_relu, 1, 1, 512, 1, 1, biased=False, relu=False, name="conv4_1_1_1_proj")
    conv4_1_1_1_proj_bn = bn(conv4_1_1_1_proj, relu=False, name="conv4_1_1_1_proj_bn", is_test=is_test)

    conv4_1_1_1_reduce = conv(conv3_4_relu, 1, 1, 128, 1, 1, biased=False, relu=False, name="conv4_1_1_1_reduce")
    conv4_1_1_1_reduce_bn = bn(conv4_1_1_1_reduce, relu=True, name="conv4_1_1_1_reduce_bn", is_test=is_test)
    conv4_1_1_1_reduce_bn_pad = zero_padding(conv4_1_1_1_reduce_bn, padding=2)
    conv4_1_3_3 = atrous_conv(conv4_1_1_1_reduce_bn_pad, 3, 3, 128, 2, biased=False, relu=False, name="conv4_1_3_3")
    conv4_1_3_3_bn = bn(conv4_1_3_3, relu=True, name="conv4_1_3_3_bn", is_test=is_test)
    conv4_1_1_1_increase = conv(conv4_1_3_3_bn, 1, 1, 512, 1, 1, biased=False, relu=False, name="conv4_1_1_1_increase")
    conv4_1_1_1_increase_bn = bn(conv4_1_1_1_increase, relu=False, name="conv4_1_1_1_increase_bn", is_test=is_test)

    conv4_1 = conv4_1_1_1_proj_bn + conv4_1_1_1_increase_bn
    conv4_1_relu = fluid.layers.relu(conv4_1, name="conv4_1_relu")
    conv4_2_1_1_reduce = conv(conv4_1_relu, 1, 1, 128, 1, 1, biased=False, relu=False, name="conv4_2_1_1_reduce")
    conv4_2_1_1_reduce_bn = bn(conv4_2_1_1_reduce, relu=True, name="conv4_2_1_1_reduce_bn", is_test=is_test)
    conv4_2_1_1_reduce_bn_pad = zero_padding(conv4_2_1_1_reduce_bn, padding=2)
    conv4_2_3_3 = atrous_conv(conv4_2_1_1_reduce_bn_pad, 3, 3, 128, 2, biased=False, relu=False, name="conv4_2_3_3")
    conv4_2_3_3_bn = bn(conv4_2_3_3, relu=True, name="conv4_2_3_3_bn", is_test=is_test)
    conv4_2_1_1_increase = conv(conv4_2_3_3_bn, 1, 1, 512, 1, 1, biased=False, relu=False, name="conv4_2_1_1_increase")
    conv4_2_1_1_increase_bn = bn(conv4_2_1_1_increase, relu=False, name="conv4_2_1_1_increase_bn", is_test=is_test)

    conv4_2 = conv4_1_relu + conv4_2_1_1_increase_bn
    conv4_2_relu = fluid.layers.relu(conv4_2, name="conv4_2_relu")
    conv4_3_1_1_reduce = conv(conv4_2_relu, 1, 1, 128, 1, 1, biased=False, relu=False, name="conv4_3_1_1_reduce")
    conv4_3_1_1_reduce_bn = bn(conv4_3_1_1_reduce, relu=True, name="conv4_3_1_1_reduce_bn", is_test=is_test)
    conv4_3_1_1_reduce_bn_pad = zero_padding(conv4_3_1_1_reduce_bn, padding=2)
    conv4_3_3_3 = atrous_conv(conv4_3_1_1_reduce_bn_pad, 3, 3, 128, 2, biased=False, relu=False, name="conv4_3_3_3")
    conv4_3_3_3_bn = bn(conv4_3_3_3, relu=True, name="conv4_3_3_3_bn", is_test=is_test)
    conv4_3_1_1_increase = conv(conv4_3_3_3_bn, 1, 1, 512, 1, 1, biased=False, relu=False, name="conv4_3_1_1_increase")
    conv4_3_1_1_increase_bn = bn(conv4_3_1_1_increase, relu=False, name="conv4_3_1_1_increase_bn", is_test=is_test)

    conv4_3 = conv4_2_relu + conv4_3_1_1_increase_bn
    conv4_3_relu = fluid.layers.relu(conv4_3, name="conv4_3_relu")
    conv4_4_1_1_reduce = conv(conv4_3_relu, 1, 1, 128, 1, 1, biased=False, relu=False, name="conv4_4_1_1_reduce")
    conv4_4_1_1_reduce_bn = bn(conv4_4_1_1_reduce, relu=True, name="conv4_4_1_1_reduce_bn", is_test=is_test)
    conv4_4_1_1_reduce_bn_pad = zero_padding(conv4_4_1_1_reduce_bn, padding=2)
    conv4_4_3_3 = atrous_conv(conv4_4_1_1_reduce_bn_pad, 3, 3, 128, 2, biased=False, relu=False, name="conv4_4_3_3")
    conv4_4_3_3_bn = bn(conv4_4_3_3, relu=True, name="conv4_4_3_3_bn", is_test=is_test)
    conv4_4_1_1_increase = conv(conv4_4_3_3_bn, 1, 1, 512, 1, 1, biased=False, relu=False, name="conv4_4_1_1_increase")
    conv4_4_1_1_increase_bn = bn(conv4_4_1_1_increase, relu=False, name="conv4_4_1_1_increase_bn", is_test=is_test)

    conv4_4 = conv4_3_relu + conv4_4_1_1_increase_bn
    conv4_4_relu = fluid.layers.relu(conv4_4, name="conv4_4_relu")
    conv4_5_1_1_reduce = conv(conv4_4_relu, 1, 1, 128, 1, 1, biased=False, relu=False, name="conv4_5_1_1_reduce")
    conv4_5_1_1_reduce_bn = bn(conv4_5_1_1_reduce, relu=True, name="conv4_5_1_1_reduce_bn", is_test=is_test)
    conv4_5_1_1_reduce_bn_pad = zero_padding(conv4_5_1_1_reduce_bn, padding=2)
    conv4_5_3_3 = atrous_conv(conv4_5_1_1_reduce_bn_pad, 3, 3, 128, 2, biased=False, relu=False, name="conv4_5_3_3")
    conv4_5_3_3_bn = bn(conv4_5_3_3, relu=True, name="conv4_5_3_3_bn", is_test=is_test)
    conv4_5_1_1_increase = conv(conv4_5_3_3_bn, 1, 1, 512, 1, 1, biased=False, relu=False, name="conv4_5_1_1_increase")
    conv4_5_1_1_increase_bn = bn(conv4_5_1_1_increase, relu=False, name="conv4_5_1_1_increase_bn", is_test=is_test)

    conv4_5 = conv4_4_relu + conv4_5_1_1_increase_bn
    conv4_5_relu = fluid.layers.relu(conv4_5, name="conv4_5_relu")
    conv4_6_1_1_reduce = conv(conv4_5_relu, 1, 1, 128, 1, 1, biased=False, relu=False, name="conv4_6_1_1_reduce")
    conv4_6_1_1_reduce_bn = bn(conv4_6_1_1_reduce, relu=True, name="conv4_6_1_1_reduce_bn", is_test=is_test)
    conv4_6_1_1_reduce_bn_pad = zero_padding(conv4_6_1_1_reduce_bn, padding=2)
    conv4_6_3_3 = atrous_conv(conv4_6_1_1_reduce_bn_pad, 3, 3, 128, 2, biased=False, relu=False, name="conv4_6_3_3")
    conv4_6_3_3_bn = bn(conv4_6_3_3, relu=True, name="conv4_6_3_3_bn", is_test=is_test)
    conv4_6_1_1_increase = conv(conv4_6_3_3_bn, 1, 1, 512, 1, 1, biased=False, relu=False, name="conv4_6_1_1_increase")
    conv4_6_1_1_increase_bn = bn(conv4_6_1_1_increase, relu=False, name="conv4_6_1_1_increase_bn", is_test=is_test)

    conv4_6 = conv4_5_relu + conv4_6_1_1_increase_bn
    conv4_6_relu = fluid.layers.relu(conv4_6, name="conv4_6_relu")
    conv5_1_1_1_proj = conv(conv4_6_relu, 1, 1, 1024, 1, 1, biased=False, relu=False, name="conv5_1_1_1_proj")
    conv5_1_1_1_proj_bn = bn(conv5_1_1_1_proj, relu=False, name="conv5_1_1_1_proj_bn", is_test=is_test)

    conv5_1_1_1_reduce = conv(conv4_6_relu, 1, 1, 256, 1, 1, biased=False, relu=False, name="conv5_1_1_1_reduce")
    conv5_1_1_1_reduce_bn = bn(conv5_1_1_1_reduce, relu=True, name="conv5_1_1_1_reduce_bn", is_test=is_test)
    conv5_1_1_1_reduce_bn_pad = zero_padding(conv5_1_1_1_reduce_bn,  padding=4)
    conv5_1_3_3 = atrous_conv(conv5_1_1_1_reduce_bn_pad, 3, 3, 256, 4, biased=False, relu=False, name="conv5_1_3_3")
    conv5_1_3_3_bn = bn(conv5_1_3_3, relu=True, name="conv5_1_3_3_bn", is_test=is_test)
    conv5_1_1_1_increase = conv(conv5_1_3_3_bn, 1, 1, 1024, 1, 1, biased=False, relu=False, name="conv5_1_1_1_increase")
    conv5_1_1_1_increase_bn = bn(conv5_1_1_1_increase, relu=False, name="conv5_1_1_1_increase_bn", is_test=is_test)

    conv5_1 = conv5_1_1_1_proj_bn + conv5_1_1_1_increase_bn
    conv5_1_relu = fluid.layers.relu(conv5_1, name="conv5_1_relu")
    conv5_2_1_1_reduce = conv(conv5_1_relu, 1, 1, 256, 1, 1, biased=False, relu=False, name="conv5_2_1_1_reduce")
    conv5_2_1_1_reduce_bn = bn(conv5_2_1_1_reduce, relu=True, name="conv5_2_1_1_reduce_bn", is_test=is_test)
    conv5_2_1_1_reduce_bn_pad = zero_padding(conv5_2_1_1_reduce_bn, padding=4)
    conv5_2_3_3 = atrous_conv(conv5_2_1_1_reduce_bn_pad, 3, 3, 256, 4, biased=False, relu=False, name="conv5_2_3_3")
    conv5_2_3_3_bn = bn(conv5_2_3_3, relu=True, name="conv5_2_3_3_bn", is_test=is_test)
    conv5_2_1_1_increase = conv(conv5_2_3_3_bn, 1, 1, 1024, 1, 1, biased=False, relu=False, name="conv5_2_1_1_increase")
    conv5_2_1_1_increase_bn = bn(conv5_2_1_1_increase, relu=False, name="conv5_2_1_1_increase_bn", is_test=is_test)

    conv5_2 = conv5_1_relu + conv5_2_1_1_increase_bn
    conv5_2_relu = fluid.layers.relu(conv5_2, name="conv5_2_relu")
    

    conv5_3_1_1_reduce = conv(conv5_2_relu, 1, 1, 256, 1, 1, biased=False, relu=False, name="conv5_3_1_1_reduce")
    conv5_3_1_1_reduce_bn = bn(conv5_3_1_1_reduce, relu=True, name="conv5_3_1_1_reduce_bn", is_test=is_test)
    conv5_3_1_1_reduce_bn_pad = zero_padding(conv5_3_1_1_reduce_bn, padding=4) 
    conv5_3_3_3 = atrous_conv(conv5_3_1_1_reduce_bn_pad, 3, 3, 256, 4, biased=False, relu=False, name="conv5_3_3_3")
    conv5_3_3_3_bn = bn(conv5_3_3_3, relu=True, name="conv5_3_3_3_bn", is_test=is_test)
    conv5_3_1_1_increase = conv(conv5_3_3_3_bn, 1, 1, 1024, 1, 1, biased=False, relu=False, name="conv5_3_1_1_increase")
    conv5_3_1_1_increase_bn = bn(conv5_3_1_1_increase, relu=False, name="conv5_3_1_1_increase_bn", is_test=is_test)

    conv5_3 = conv5_2_relu + conv5_3_1_1_increase_bn
    conv5_3_relu = fluid.layers.relu(conv5_3)
    

    shape = np.ceil(input_shape / 32).astype("int32")
    h, w = shape

    conv5_3_pool1 = avg_pool(conv5_3_relu, h, w, h, w)
    conv5_3_pool1_interp = interp(conv5_3_pool1, shape)

    conv5_3_pool2 = avg_pool(conv5_3_relu, h/2, w/2, h/2, w/2)
    conv5_3_pool2_interp = interp(conv5_3_pool2, shape)

    conv5_3_pool3 = avg_pool(conv5_3_relu, h/3, w/3, h/3, w/3)
    conv5_3_pool3_interp = interp(conv5_3_pool3, shape)

    conv5_3_pool6 = avg_pool(conv5_3_relu, h/4, w/4, h/4, w/4)
    conv5_3_pool6_interp = interp(conv5_3_pool6, shape)

    conv5_3_sum = conv5_3_relu + conv5_3_pool6_interp + conv5_3_pool3_interp + conv5_3_pool2_interp + conv5_3_pool1_interp
    conv5_4_k1 = conv(conv5_3_sum, 1, 1, 256, 1, 1, biased=False, relu=False, name="conv5_4_k1")
    conv5_4_k1_bn = bn(conv5_4_k1, relu=True, name="conv5_4_k1_bn", is_test=is_test)
    conv5_4_interp = interp(conv5_4_k1_bn, input_shape/16)
    conv5_4_interp_pad = zero_padding(conv5_4_interp, padding=2)
    conv_sub4 = atrous_conv(conv5_4_interp_pad, 3, 3, 128, 2, biased=False, relu=False, name="conv_sub4")
    conv_sub4_bn = bn(conv_sub4, relu=False, name="conv_sub4_bn", is_test=is_test)

    conv3_1_sub2_proj = conv(conv3_1_relu, 1, 1, 128, 1, 1, biased=False, relu=False, name="conv3_1_sub2_proj")
    conv3_1_sub2_proj_bn = bn(conv3_1_sub2_proj, relu=False, name="conv3_1_sub2_proj_bn", is_test=is_test)

    sub24_sum = conv_sub4_bn + conv3_1_sub2_proj_bn
    sub24_sum_relu = fluid.layers.relu(sub24_sum)
    sub24_sum_interp = interp(sub24_sum_relu, input_shape / 8)


    sub24_sum_interp_pad = zero_padding(sub24_sum_interp, padding=2)
    conv_sub2 = atrous_conv(sub24_sum_interp_pad, 3, 3, 128, 2, biased=False, relu=False, name="conv_sub2")
    conv_sub2_bn = bn(conv_sub2, relu=False, name="conv_sub2_bn", is_test=is_test)

    
    conv1_sub1 = conv(data, 3, 3, 32, 2, 2, biased=False, padding='SAME', relu=False, name="conv1_sub1")
    conv1_sub1_bn = bn(conv1_sub1, relu=True, name="conv1_sub1_bn", is_test=is_test)
    conv2_sub1 = conv(conv1_sub1_bn, 3, 3, 32, 2, 2, biased=False, padding='SAME', relu=False, name="conv2_sub1")
    conv2_sub1_bn = bn(conv2_sub1, relu=True, name="conv2_sub1_bn", is_test=is_test)
    conv3_sub1 = conv(conv2_sub1_bn, 3, 3, 64, 2, 2, biased=False, padding='SAME', relu=False, name="conv3_sub1")
    conv3_sub1_bn = bn(conv3_sub1, relu=True, name="conv3_sub1_bn", is_test=is_test)
    conv3_sub1_proj = conv(conv3_sub1_bn, 1, 1, 128, 1, 1, biased=False, relu=False, name="conv3_sub1_proj")
    conv3_sub1_proj_bn = bn(conv3_sub1_proj, relu=False, name="conv3_sub1_proj_bn", is_test=is_test)


    sub12_sum = conv_sub2_bn + conv3_sub1_proj_bn
    sub12_sum_relu = fluid.layers.relu(sub12_sum)
    sub12_sum_interp = interp(sub12_sum_relu, input_shape / 4)
    conv6_cls = conv(sub12_sum_interp, 1, 1, num_classes, 1, 1, biased=True, relu=False, name="conv6_cls")
    
    sub4_out = conv(conv5_4_interp, 1, 1, num_classes, 1, 1, biased=True, relu=False, name="sub4_out")
    
    sub24_out = conv(sub24_sum_interp, 1, 1, num_classes, 1, 1, biased=True, relu=False, name="sub24_out")
    return sub4_out, sub24_out, conv6_cls
