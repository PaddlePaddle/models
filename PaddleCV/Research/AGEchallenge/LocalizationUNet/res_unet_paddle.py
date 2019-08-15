import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as FL
from paddle.fluid.param_attr import ParamAttr

from resnet import *

def conv_bn_layer(input,
                  num_filters,
                  filter_size,
                  stride=1,
                  groups=1,
                  act=None,
                  name=None):
    conv = FL.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) // 2,
        groups=groups,
        act=None,
        param_attr=ParamAttr(name=name + "_weights"),
        bias_attr=False,
        name=name + '.conv2d.output.1')
    
    bn_name = name + "_bn"
    return FL.batch_norm(input=conv, 
                                   act=act,
                                   name=bn_name+'.output.1',
                                   param_attr=ParamAttr(name=bn_name + '_scale'),
                                   bias_attr=ParamAttr(bn_name + '_offset'),
                                   moving_mean_name=bn_name + '_mean',
                                   moving_variance_name=bn_name + '_variance',)

def DoubleConv_up(x, out_channels, name=None):
    x = conv_bn_layer(x, out_channels, 3, 1, act='relu', name=name+"1")
    x = conv_bn_layer(x, out_channels, 3, 1, act='relu', name=name+"2")
    return x


def ConvUp(x1, x2, out_channels, name=None):
    x1 = FL.conv2d_transpose(x1, num_filters=x1.shape[1] // 2, filter_size=2, stride=2)
    x = FL.concat([x1,x2], axis=1)
    x = DoubleConv_up(x, out_channels, name=name+"_doubleconv")
    return x

class ResUNet():
    def __init__(self, backbone, out_channels):
        self.backbone = backbone
        self.out_channels = out_channels

    def net(self, input):
        c1, c2, c3, c4, c5 = self.backbone(input)
        channels = [64, 128, 256, 512]
        x = ConvUp(c5, c4, channels[2], name='up5')
        x = ConvUp(x, c3, channels[1], name='up6')
        x = ConvUp(x, c2, channels[0], name='up7')
        x = ConvUp(x, c1, channels[0], name='up8')
        x = FL.conv2d_transpose(x, num_filters=self.out_channels, filter_size=2, stride=2)

        return x