import os

import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn
from ltr.admin.environment import env_settings

CURRENT_DIR = os.path.dirname(__file__)


def weight_init():
    init = fluid.initializer.MSRAInitializer(uniform=False)
    param = fluid.ParamAttr(initializer=init)
    return param


def norm_weight_init(constant=1.0):
    init = fluid.initializer.ConstantInitializer(constant)
    param = fluid.ParamAttr(initializer=init)
    return param


def norm_bias_init():
    init = fluid.initializer.ConstantInitializer(value=0.)
    param = fluid.ParamAttr(initializer=init)
    return param


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 filter_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bn_init_constant=1.0):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            num_channels=in_channels,
            filter_size=filter_size,
            num_filters=out_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            param_attr=weight_init(),
            bias_attr=False)
        self.bn = nn.BatchNorm(
            out_channels,
            param_attr=norm_weight_init(bn_init_constant),
            bias_attr=norm_bias_init(),
            act=None,
            momentum=0.1,
            use_global_stats=True)

    def forward(self, inputs):
        res = self.conv(inputs)
        self.conv_res = res
        res = self.bn(res)
        return res


class BasicBlock(fluid.dygraph.Layer):
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 is_downsample=None):

        super(BasicBlock, self).__init__()

        self.expansion = 1

        self.conv_bn1 = ConvBNLayer(
            num_channels=in_channels,
            out_channels=out_channels,
            filter_size=3,
            stride=stride,
            groups=1)
        self.conv_bn2 = ConvBNLayer(
            out_channels=out_channels,
            filter_size=3,
            stride=1,
            groups=1)

        self.is_downsample = is_downsample
        if self.is_downsample:
            self.downsample = ConvBNLayer(
                num_channels=in_channels,
                out_channels=out_channels,
                filter_size=1,
                stride=stride)

        self.stride = stride

    def forward(self, inputs):
        identity = inputs
        res = self.conv_bn1(inputs)
        res = fluid.layers.relu(res)

        res = self.conv_bn2(res)

        if self.is_downsample:
            identity = self.downsample(identity)

        res += identity
        res = fluid.layers.relu(res)
        return res


class Bottleneck(fluid.dygraph.Layer):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 base_width=64,
                 dilation=1,
                 groups=1):
        super(Bottleneck, self).__init__()

        width = int(out_channels*(base_width / 64.))*groups

        self.conv_bn1 = ConvBNLayer(
            in_channels=in_channels,
            filter_size=1,
            out_channels=width,
            groups=1)

        padding = 2 - stride
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation

        assert stride == 1 or dilation == 1, \
            "stride and dilation must have one equals to zero at least"

        if dilation > 1:
            padding = dilation

        self.conv_bn2 = ConvBNLayer(
            in_channels=width,
            filter_size=3,
            out_channels=width,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups)
        self.conv_bn3 = ConvBNLayer(
            in_channels=width,
            filter_size=1,
            out_channels=out_channels*self.expansion,
            bn_init_constant=0.)

        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs):
        identify = inputs

        out = self.conv_bn1(inputs)
        out = fluid.layers.relu(out)

        out = self.conv_bn2(out)
        out = fluid.layers.relu(out)

        out = self.conv_bn3(out)

        if self.downsample is not None:
            identify = self.downsample(inputs)

        out += identify
        out = fluid.layers.relu(out)
        return out


class ResNet(fluid.dygraph.Layer):
    def __init__(self, name, Block, layers, output_layers, is_test=False):
        """

        :param name: str, namescope
        :param layers: int, the layer of defined network
        :param output_layers: list of int, the layers for output
        """
        super(ResNet, self).__init__(name_scope=name)

        support_layers = [50]
        assert layers in support_layers, \
        "support layer can only be one of [50, ]"
        self.layers = layers
        self.feat_layers = ['block{}'.format(i) for i in output_layers]
        output_depth = max(output_layers) + 1
        self.is_test = is_test

        if layers == 18:
            depths = [2, 2, 2, 2]
        elif layers == 50 or layers == 34:
            depths = [3, 4, 6, 3]
        elif layers == 101:
            depths = [3, 4, 23, 3]
        elif layers == 152:
            depths = [3, 8, 36, 3]

        strides = [1, 2, 1, 1]
        num_filters = [64, 128, 256, 512]
        dilations = [1, 1, 2, 4]

        self.in_channels = 64
        self.dilation = 1

        self.conv_bn_init = ConvBNLayer(
            in_channels=3,
            out_channels=self.in_channels,
            filter_size=7,
            stride=2)

        self.maxpool = nn.Pool2D(
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type="max")

        block_collect = []
        downsample = None
        for i in range(min(len(depths), output_depth)):
            # collect layers in each block
            _block = []

            stride = strides[i]
            out_channel = num_filters[i]
            dilation = dilations[i]

            if stride != 1 or self.in_channels != self.in_channels*Block.expansion:
                if stride == 1 and dilation == 1:
                    downsample = ConvBNLayer(
                        in_channels=self.in_channels,
                        out_channels=out_channel*Block.expansion,
                        filter_size=1,
                        stride=stride)
                else:
                    if dilation > 1:
                        dd = dilation // 2
                        padding = dd
                    else:
                        dd = 1
                        padding = 0
                    downsample = ConvBNLayer(
                        in_channels=self.in_channels,
                        out_channels=out_channel*Block.expansion,
                        filter_size=3,
                        stride=stride,
                        padding=padding,
                        dilation=dd)

            bottleneck_block = self.add_sublayer(
                "block{}_0".format(i),
                Block(
                    in_channels=self.in_channels,
                    out_channels=out_channel,
                    stride=stride,
                    dilation=dilation,
                    downsample=downsample))

            _block.append(bottleneck_block)

            self.in_channels = num_filters[i]*Block.expansion

            for j in range(1, depths[i]):
                bottleneck_block = self.add_sublayer(
                    "block{}_{}".format(i, j),
                    Block(self.in_channels, out_channel, dilation=dilation))
                _block.append(bottleneck_block)

            # collect blocks
            block_collect.append(_block)

        self.block_collect = block_collect

    @fluid.dygraph.no_grad
    def forward(self, inputs):
        out = []
        res = self.conv_bn_init(inputs)
        res = fluid.layers.relu(res)
        out.append(res)
        res = self.maxpool(res)
        for i in range(len(self.block_collect)):

            for layer in self.block_collect[i]:
                res = layer(res)

            name = 'block{}'.format(i)
            if name in self.feat_layers:
                out.append(res)
                if (len(out) - 1) == len(self.feat_layers):
                    out[-1].stop_gradient = True if self.is_test else False
                    if len(out) == 1:
                      return out[0]
                    else:
                      return out

        out[-1].stop_gradient = True if self.is_test else False
        return out


def resnet50(name, pretrained=False, **kwargs):
    net = ResNet(name, Block=Bottleneck, layers=50, **kwargs)
    if pretrained:
        params_path = os.path.join(env_settings().backbone_dir, 'ResNet50_dilated')
        print("=> loading backbone model from '{}'".format(params_path))
        params, _ = fluid.load_dygraph(params_path)
        net.load_dict(params)
        print("Done")
        
    return net
