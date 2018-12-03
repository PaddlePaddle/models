import paddle
import paddle.fluid as fluid
import math
__all__ = ["VGGNet", "VGG11", "VGG13", "VGG16", "VGG19"]

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.01, 0.01, 0.001, 0.0001]
    }
}


class VGGNet():
    def __init__(self, layers=16):
        self.params = train_parameters
        self.layers = layers

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
        conv1 = self.conv_block(input, 64, nums[0])
        conv2 = self.conv_block(conv1, 128, nums[1])
        conv3 = self.conv_block(conv2, 256, nums[2])
        conv4 = self.conv_block(conv3, 512, nums[3])
        conv5 = self.conv_block(conv4, 512, nums[4])
        fc_dim = 4096

        fc6 = fluid.layers.fc(input=conv5,
                              size=fc_dim,
                              act='relu',
                              bias_attr=fluid.param_attr.ParamAttr(),
                              param_attr=fluid.param_attr.ParamAttr())
        drop6 = fluid.layers.dropout(x=fc6, dropout_prob=0.5)

        fc7 = fluid.layers.fc(input=drop6,
                              size=fc_dim,
                              act='relu',
                              bias_attr=fluid.param_attr.ParamAttr(),
                              param_attr=fluid.param_attr.ParamAttr())
        drop7 = fluid.layers.dropout(x=fc7, dropout_prob=0.5)

        out = fluid.layers.fc(input=drop7,
                              size=class_dim,
                              act='softmax',
                              bias_attr=fluid.param_attr.ParamAttr(),
                              param_attr=fluid.param_attr.ParamAttr())
        return out

    def conv_block(self, input, num_filter, groups):
        conv = input
        for i in range(groups):
            conv = fluid.layers.conv2d(
                input=conv,
                num_filters=num_filter,
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',
                param_attr=fluid.param_attr.ParamAttr(),
                bias_attr=fluid.param_attr.ParamAttr())
            pool = fluid.layers.pool2d(
                input=conv, pool_size=2, pool_type='max', pool_stride=2)
        return pool


def VGG11():
    model = VGGNet(layers=11)
    return model


def VGG13():
    model = VGGNet(layers=13)
    return model


def VGG16():
    model = VGGNet(layers=16)
    return model


def VGG19():
    model = VGGNet(layers=19)
    return model
