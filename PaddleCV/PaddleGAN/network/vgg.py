from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid

__all__ = ["VGGNet", "VGG11", "VGG13", "VGG16", "VGG19"]

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


class VGGNet():
    def __init__(self, layers=16, name=""):
        self.params = train_parameters
        self.layers = layers
        self.name=name

    def net(self, input, class_dim=1000):
        layers = self.layers
        vgg_spec = {
            11: ([1, 1, 2, 2, 2]),
            13: ([2, 2, 2, 2, 2]),
            16: ([2, 2, 3, 3, 3]),
            19: ([2, 2, 4, 4, 4])
        }
        assert layers in vgg_spec.keys(), \
            "supported layers are {} but input layer is {}".format(vgg_spec.keys(), layers)

        nums = vgg_spec[layers]
        conv1, res = self.conv_block(input, 64, nums[0], name=self.name+"_conv1_")
        conv2, res = self.conv_block(res, 128, nums[1], name=self.name+"_conv2_")
        conv3, res = self.conv_block(res, 256, nums[2], name=self.name+"_conv3_")
        conv4, res = self.conv_block(res, 512, nums[3], name=self.name+"_conv4_")
        conv5, res = self.conv_block(res, 512, nums[4], name=self.name+"_conv5_")

        if self.layers == 16:
            return [conv1, conv2, conv3]
        elif self.layers == 19:
            return [conv1, conv2, conv3, conv4, conv5]

    def conv_block(self, input, num_filter, groups, name=""):
        conv = input
        for i in range(groups):
            conv = fluid.layers.conv2d(
                input=conv,
                num_filters=num_filter,
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',
                param_attr=fluid.param_attr.ParamAttr(
                    name=name + str(i + 1) + "_weights", trainable=False),
                bias_attr=False
                )
            if i == 0:
                relu_res = conv
        return relu_res, fluid.layers.pool2d(
            input=conv, pool_size=2, pool_type='max', pool_stride=2)


    def load_vars(self, exe, program, pretrained_model):
        vars = []
        for var in program.list_vars():
            if fluid.io.is_parameter(var) and var.name.startswith("vgg"):
                vars.append(var)
                print(var.name)
        fluid.io.load_vars(exe, pretrained_model, program, vars)


def VGG11():
    model = VGGNet(layers=11)
    return model


def VGG13():
    model = VGGNet(layers=13)
    return model


def VGG16():
    model = VGGNet(layers=16, name="vgg16")
    return model


def VGG19(name="vgg19"):
    model = VGGNet(layers=19, name=name)
    return model
