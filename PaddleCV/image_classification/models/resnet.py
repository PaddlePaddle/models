from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid
import math
from paddle.fluid.param_attr import ParamAttr

__all__ = ["ResNet", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}


class ResNet():
    def __init__(self, layers=50):
        self.params = train_parameters
        self.layers = layers

    def net(self, input, class_dim=1000):
        layers = self.layers
        supported_layers = [18, 34, 50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(
            input=input, num_filters=64, filter_size=7, stride=2, act='relu',name="conv1")
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
        if layers >= 50:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name="res"+str(block+2)+"a"
                        else:
                            conv_name="res"+str(block+2)+"b"+str(i)
                    else:
                        conv_name="res"+str(block+2)+chr(97+i)
                    conv = self.bottleneck_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1, name=conv_name)

            pool = fluid.layers.pool2d(
                input=conv, pool_size=7, pool_type='avg', global_pooling=True)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = fluid.layers.fc(input=pool,
                                  size=class_dim,
                                  param_attr=fluid.param_attr.ParamAttr(
                                      initializer=fluid.initializer.Uniform(-stdv, stdv)))
        else:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    conv_name="res"+str(block+2)+chr(97+i)
                    conv = self.basic_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        is_first=block==i==0,
                        name=conv_name)

            pool = fluid.layers.pool2d(
                input=conv, pool_size=7, pool_type='avg', global_pooling=True)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = fluid.layers.fc(input=pool,
                                  size=class_dim,
                                  param_attr=fluid.param_attr.ParamAttr(
                                      initializer=fluid.initializer.Uniform(-stdv, stdv)))
        return out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        conv = fluid.layers.conv2d(
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
        
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:] 
        return fluid.layers.batch_norm(input=conv, 
                                       act=act,
                                       name=bn_name+'.output.1',
                                       param_attr=ParamAttr(name=bn_name + '_scale'),
                                       bias_attr=ParamAttr(bn_name + '_offset'),
                                       moving_mean_name=bn_name + '_mean',
                                       moving_variance_name=bn_name + '_variance',)

    def shortcut(self, input, ch_out, stride, is_first, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1 or is_first == True:
            return self.conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name):
        conv0 = self.conv_bn_layer(
            input=input, num_filters=num_filters, filter_size=1, act='relu',name=name+"_branch2a")
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
        name=name+"_branch2b")
        conv2 = self.conv_bn_layer(
            input=conv1, num_filters=num_filters * 4, filter_size=1, act=None, name=name+"_branch2c")

        short = self.shortcut(input, num_filters * 4, stride, is_first=False, name=name + "_branch1")

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu',name=name+".add.output.5")
    
    def basic_block(self, input, num_filters, stride, is_first, name):
        conv0 = self.conv_bn_layer(input=input, num_filters=num_filters, filter_size=3, act='relu', stride=stride,
                                   name=name+"_branch2a")
        conv1 = self.conv_bn_layer(input=conv0, num_filters=num_filters, filter_size=3, act=None, 
                                   name=name+"_branch2b")
        short = self.shortcut(input, num_filters, stride, is_first, name=name + "_branch1")
        return fluid.layers.elementwise_add(x=short, y=conv1, act='relu')
    

def ResNet18():
    model = ResNet(layers=18)
    return model


def ResNet34():
    model = ResNet(layers=34)
    return model


def ResNet50():
    model = ResNet(layers=50)
    return model


def ResNet101():
    model = ResNet(layers=101)
    return model


def ResNet152():
    model = ResNet(layers=152)
    return model
