import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn
import os.path as osp
import sys

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


class AdjustLayer(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, is_test=False):
        super(AdjustLayer, self).__init__()
        self.conv = nn.Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1, 
            param_attr=weight_init(),
            bias_attr=False)
        self.bn = nn.BatchNorm(
            num_channels=num_filters,
            param_attr=norm_weight_init(),
            bias_attr=norm_bias_init(),
            momentum=0.9,
            act=None,
            use_global_stats=is_test)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if x.shape[3] < 20:
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]
        return x


class AdjustAllLayer(fluid.dygraph.Layer):
    def __init__(self, in_channels, out_channels, is_test=False):
        super(AdjustAllLayer, self).__init__('')
        self.num = len(out_channels)
        self.sub_layer_list = []
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0], is_test)
        else:
            for i in range(self.num):
                Build_Adjust_Layer = self.add_sublayer(
                    'downsample'+str(i+2),
                    AdjustLayer(in_channels[i], out_channels[i], is_test))
                self.sub_layer_list.append(Build_Adjust_Layer)

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                build_adjust_layer_i = sub_layer_list[i]
                out.append(build_adjust_layer_i(features[i]))
            return out
