import os

import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn
from ltr_pp.admin.environment import env_settings

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
    def __init__(self, in_channels, out_channels, filter_size, stride=1, groups=1,
                 bn_init_constant=1.0, is_test=False):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(num_channels=in_channels, filter_size=filter_size,
                              num_filters=out_channels,
                              stride=stride, padding=(filter_size - 1) // 2,
                              groups=groups, bias_attr=False,
                              param_attr=weight_init())
        self.bn = nn.BatchNorm(out_channels,
                               param_attr=norm_weight_init(bn_init_constant),
                               bias_attr=norm_bias_init(),
                               act=None, momentum=0.9,
                               use_global_stats=is_test)

    def forward(self, inputs):
        res = self.conv(inputs)
        self.conv_res = res
        res = self.bn(res)
        return res


class BasicBlock(fluid.dygraph.Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels,
                 stride=1, is_downsample=None, is_test=False):

        super(BasicBlock, self).__init__()
        self.expansion = 1

        self.conv_bn1 = ConvBNLayer(in_channels=in_channels, out_channels=out_channels,
                                    filter_size=3, stride=stride, groups=1, is_test=is_test)
        self.conv_bn2 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels,
                                    filter_size=3, stride=1, groups=1, is_test=is_test)

        self.is_downsample = is_downsample
        if self.is_downsample:
            self.downsample = ConvBNLayer(in_channels=in_channels,
                                          out_channels=out_channels,
                                          filter_size=1, stride=stride, is_test=is_test)
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

    def __init__(self, in_channels, out_channels, stride=1, is_downsample=None,
                 base_width=64, dilation=1, groups=1, is_test=False):
        super(Bottleneck, self).__init__()

        width = int(out_channels * (base_width / 64.)) * groups

        self.conv_bn1 = ConvBNLayer(in_channels=in_channels, filter_size=1, out_channels=width, groups=1,
                                    is_test=is_test)
        self.conv_bn2 = ConvBNLayer(in_channels=width, filter_size=3, out_channels=width, stride=stride,
                                    groups=groups, is_test=is_test)
        self.conv_bn3 = ConvBNLayer(in_channels=width, filter_size=1,
                                    out_channels=out_channels * self.expansion,
                                    bn_init_constant=0., is_test=is_test)
        self.is_downsample = is_downsample
        if self.is_downsample:
            self.downsample = ConvBNLayer(in_channels=in_channels,
                                          out_channels=out_channels * self.expansion,
                                          filter_size=1, stride=stride, is_test=is_test)

        self.stride = stride

    def forward(self, inputs):
        identify = inputs

        out = self.conv_bn1(inputs)
        out = fluid.layers.relu(out)

        out = self.conv_bn2(out)
        out = fluid.layers.relu(out)

        out = self.conv_bn3(out)

        if self.is_downsample:
            identify = self.downsample(inputs)

        out += identify
        out = fluid.layers.relu(out)
        return out


class ResNet(fluid.dygraph.Layer):
    def __init__(self, name, Block, layers, num_classes=1000,
                 groups=1, is_test=False):
        """

        :param name: str, namescope
        :param layers: int, the layer of defined network
        :param num_classes: int, the dimension of final output
        :param groups: int, default is 1
        """
        super(ResNet, self).__init__(name_scope=name)

        support_layers = [18, 34, 50, 101, 152]
        assert layers in support_layers, \
            "support layer can only be one of [18, 34, 50, 101, 152]"
        self.layers = layers

        if layers == 18:
            depths = [2, 2, 2, 2]
        elif layers == 50 or layers == 34:
            depths = [3, 4, 6, 3]
        elif layers == 101:
            depths = [3, 4, 23, 3]
        elif layers == 152:
            depths = [3, 8, 36, 3]

        strides = [1, 2, 2, 2]
        num_filters = [64, 128, 256, 512]

        self.in_channels = 64
        self.dilation = 1
        self.groups = groups

        self.conv_bn_init = ConvBNLayer(3, out_channels=self.in_channels,
                                        filter_size=7, stride=2, is_test=is_test)

        block_collect = []
        downsample = None
        for i in range(len(depths)):
            # collect layers in each block
            _block = []

            stride = strides[i]
            out_channel = num_filters[i]

            if stride != 1 or self.in_channels != num_filters[i] * Block.expansion:
                downsample = True
            bottleneck_block = self.add_sublayer("block{}_0".format(i),
                                                 Block(self.in_channels, out_channel, stride=stride,
                                                       is_downsample=downsample, is_test=is_test))

            downsample = False

            _block.append(bottleneck_block)

            self.in_channels = num_filters[i] * Block.expansion

            for j in range(1, depths[i]):
                bottleneck_block = self.add_sublayer("block{}_{}".format(i, j),
                                                     Block(self.in_channels, out_channel,
                                                           is_test=is_test))
                _block.append(bottleneck_block)

            # collect blocks
            block_collect.append(_block)

        self.block_collect = block_collect

        self.maxpool = nn.Pool2D(pool_size=3, pool_stride=2,
                                 pool_padding=1, pool_type="max")

        self.global_pool = nn.Pool2D(pool_type='avg', global_pooling=True)
        self.fc = nn.Linear(input_dim=512 * Block.expansion, output_dim=num_classes)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, inputs, feat_layers):
        out = {}
        res = self.conv_bn_init(inputs)
        res = fluid.layers.relu(res)
        res = self.maxpool(res)

        # out['conv_init'] = res
        for i in range(len(self.block_collect)):

            for layer in self.block_collect[i]:
                res = layer(res)

            name = 'block{}'.format(i)
            if name in feat_layers:
                out[name] = res
                if len(out) == len(feat_layers):
                    return out

        res = self.global_pool(res)
        B, C, _, _ = res.shape
        res = fluid.layers.reshape(res, [B, C])
        res = self.fc(res)
        out['fc'] = res
        return out


def resnet18(name, is_test=False, pretrained=False):
    net = ResNet(name, Block=BasicBlock, layers=18, is_test=is_test)
    if pretrained:
        params_path = os.path.join(env_settings().backbone_dir, 'ResNet18')
        print("=> loading backbone model from '{}'".format(params_path))
        params, _ = fluid.load_dygraph(params_path)
        net.load_dict(params)
        print("Done")
    return net


def resnet50(name, is_test=False, pretrained=False):
    net = ResNet(name, Block=Bottleneck, layers=50, is_test=is_test)
    if pretrained:
        params_path = os.path.join(env_settings().backbone_dir, 'ResNet50')
        print("=> loading backbone model from '{}'".format(params_path))
        params, _ = fluid.load_dygraph(params_path)
        net.load_dict(params)
        print("Done")
    return net
