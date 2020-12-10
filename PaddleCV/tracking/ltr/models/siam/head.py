import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn
import os.path as osp
import sys

from ltr.models.siam.xcorr import xcorr, xcorr_depthwise

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..', '..', '..'))


def weight_init():
    init = fluid.initializer.MSRAInitializer(uniform=False)
    param = fluid.ParamAttr(initializer=init)
    return param


def bias_init():
    init = fluid.initializer.ConstantInitializer(value=0.)
    param = fluid.ParamAttr(initializer=init)
    return param


def norm_weight_init():
    init = fluid.initializer.Uniform(low=0., high=1.)
    param = fluid.ParamAttr(initializer=init)
    return param


def norm_bias_init():
    init = fluid.initializer.ConstantInitializer(value=0.)
    param = fluid.ParamAttr(initializer=init)
    return param


class RPN(fluid.dygraph.Layer):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


class DepthwiseXCorr(fluid.dygraph.Layer):
    def __init__(self,
                 in_channels,
                 hidden,
                 out_channels, 
                 filter_size=3,
                 is_test=False):
        super(DepthwiseXCorr, self).__init__()
        self.kernel_conv1 = nn.Conv2D(
            num_channels=in_channels,
            num_filters=hidden,
            filter_size=filter_size,
            stride=1,
            padding=0,
            groups=1,
            param_attr=weight_init(),
            bias_attr=False)
        self.kernel_bn1 = nn.BatchNorm(
            num_channels=hidden,
            act='relu',
            param_attr=norm_weight_init(),
            bias_attr=norm_bias_init(),
            momentum=0.9,
            use_global_stats=is_test)

        self.search_conv1 = nn.Conv2D(
            num_channels=in_channels,
            num_filters=hidden,
            filter_size=filter_size,
            stride=1,
            padding=0,
            groups=1,
            param_attr=weight_init(),
            bias_attr=False)
        self.search_bn1 = nn.BatchNorm(
            num_channels=hidden,
            act='relu',
            param_attr=norm_weight_init(),
            bias_attr=norm_bias_init(),
            momentum=0.9,
            use_global_stats=is_test)

        self.head_conv1 = nn.Conv2D(
            num_channels=hidden,
            num_filters=hidden,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            param_attr=weight_init(),
            bias_attr=False)
        self.head_bn1 = nn.BatchNorm(
            num_channels=hidden,
            act='relu',
            param_attr=norm_weight_init(),
            bias_attr=norm_bias_init(),
            momentum=0.9,
            use_global_stats=is_test)
        self.head_conv2 = nn.Conv2D(
            num_channels=hidden,
            num_filters=out_channels,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            param_attr=weight_init())


    def forward(self, kernel, search):
        kernel = self.kernel_conv1(kernel)
        kernel = self.kernel_bn1(kernel)

        search = self.search_conv1(search)
        search = self.search_bn1(search)

        feature = xcorr_depthwise(search, kernel)
        out = self.head_conv1(feature)
        out = self.head_bn1(out)
        out = self.head_conv2(out)
        return out


class DepthwiseRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256, is_test=False):
        super(DepthwiseRPN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num, is_test=is_test)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num, is_test=is_test)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class MaskCorr(DepthwiseXCorr):
    def __init__(self,
                 in_channels,
                 hidden,
                 out_channels, 
                 filter_size=3,
                 hidden_filter_size=5,
                 is_test=False):
        super(MaskCorr, self).__init__(
            in_channels,
            hidden,
            out_channels, 
            filter_size,
            is_test)

    def forward(self, kernel, search):
        kernel = self.kernel_conv1(kernel)
        kernel = self.kernel_bn1(kernel)

        search = self.search_conv1(search)
        search = self.search_bn1(search)
        
        feature = xcorr_depthwise(search, kernel)
        out = self.head_conv1(feature)
        out = self.head_bn1(out)
        out = self.head_conv2(out)
        return out, feature


class RefineModule(fluid.dygraph.Layer):
    def __init__(self,
                 in_channels,
                 hidden1,
                 hidden2,
                 out_channels,
                 out_shape,
                 filter_size=3,
                 padding=1):
        super(RefineModule, self).__init__()
        self.v_conv0 = nn.Conv2D(
            num_channels=in_channels,
            num_filters=hidden1,
            filter_size=filter_size,
            stride=1,
            padding=padding,
            groups=1,
            param_attr=weight_init())
        self.v_conv1 = nn.Conv2D(
            num_channels=hidden1,
            num_filters=hidden2,
            filter_size=filter_size,
            stride=1,
            padding=padding,
            groups=1,
            param_attr=weight_init())
        self.h_conv0 = nn.Conv2D(
            num_channels=hidden2,
            num_filters=hidden2,
            filter_size=filter_size,
            stride=1,
            padding=padding,
            groups=1,
            param_attr=weight_init())
        self.h_conv1 = nn.Conv2D(
            num_channels=hidden2,
            num_filters=hidden2,
            filter_size=filter_size,
            stride=1,
            padding=padding,
            groups=1,
            param_attr=weight_init())
            
        self.out_shape = out_shape
        self.post = nn.Conv2D(
            num_channels=hidden2,
            num_filters=out_channels,
            filter_size=filter_size,
            stride=1,
            padding=padding,
            groups=1,
            param_attr=weight_init())

    def forward(self, xh, xv):
        yh = self.h_conv0(xh)
        yh = fluid.layers.relu(yh)
        yh = self.h_conv1(yh)
        yh = fluid.layers.relu(yh)

        yv = self.v_conv0(xv)
        yv = fluid.layers.relu(yv)
        yv = self.v_conv1(yv)
        yv = fluid.layers.relu(yv)

        out = yh + yv
        out = fluid.layers.resize_nearest(out, out_shape=self.out_shape, align_corners=False)
        out = self.post(out) 
        return out

class Refine(fluid.dygraph.Layer):
    def __init__(self):
        super(Refine, self).__init__()
        self.U4 = RefineModule(
            in_channels=64,
            hidden1=16,
            hidden2=4,
            out_channels=1,
            filter_size=3,
            padding=1,
            out_shape=[127, 127])

        self.U3 = RefineModule(
            in_channels=256,
            hidden1=64,
            hidden2=16,
            out_channels=4,
            filter_size=3,
            padding=1,
            out_shape=[61, 61])

        self.U2 = RefineModule(
            in_channels=512,
            hidden1=128,
            hidden2=32,
            out_channels=16,
            filter_size=3,
            padding=1,
            out_shape=[31, 31])

        self.deconv = nn.Conv2DTranspose(
            num_channels=256,
            num_filters=32,
            filter_size=15,
            padding=0,
            stride=15)


    def forward(self, xf, corr_feature, pos=None, test=False):
        if test:
            p0 = fluid.layers.pad2d(xf[0], [16, 16, 16, 16])
            p0 = p0[:, :, 4*pos[0]:4*pos[0]+61, 4*pos[1]:4*pos[1]+61]
            p1 = fluid.layers.pad2d(xf[1], [8, 8, 8, 8])
            p1 = p1[:, :, 2*pos[0]:2*pos[0]+31, 2*pos[1]:2*pos[1]+31]
            p2 = fluid.layers.pad2d(xf[2], [4, 4, 4, 4])
            p2 = p2[:, :, pos[0]:pos[0]+15, pos[1]:pos[1]+15]
            p3 = corr_feature[:, :, pos[0], pos[1]]
            p3 = fluid.layers.reshape(p3, [-1, 256, 1, 1])
        else:
            p0 = fluid.layers.unfold(xf[0], [61, 61], 4, 0)
            p0 = fluid.layers.transpose(p0, [0, 2, 1])
            p0 = fluid.layers.reshape(p0, [-1, 64, 61, 61])
            p1 = fluid.layers.unfold(xf[1], [31, 31], 2, 0)
            p1 = fluid.layers.transpose(p1, [0, 2, 1])
            p1 = fluid.layers.reshape(p1, [-1, 256, 31, 31])
            p2 = fluid.layers.unfold(xf[2], [15, 15], 1, 0)
            p2 = fluid.layers.transpose(p2, [0, 2, 1])
            p2 = fluid.layers.reshape(p2, [-1, 512, 15, 15])
            p3 = fluid.layers.transpose(corr_feature, [0, 2, 3, 1])
            p3 = fluid.layers.reshape(p3, [-1, 256, 1, 1])

        out = self.deconv(p3)
        out = self.U2(out, p2)
        out = self.U3(out, p1)
        out = self.U4(out, p0)
        out = fluid.layers.reshape(out, [-1, 127*127])

        return out
