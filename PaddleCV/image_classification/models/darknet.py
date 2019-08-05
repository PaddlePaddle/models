import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
import math
__all__ = ["DarkNet53"]

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


class DarkNet53():
    def __init__(self):
        self.params = train_parameters

    def net(self, input, class_dim=1000):
        DarkNet_cfg = {53: ([1, 2, 8, 8, 4], self.basicblock)}
        stages, block_func = DarkNet_cfg[53]
        stages = stages[0:5]
        conv1 = self.conv_bn_layer(
            input,
            ch_out=32,
            filter_size=3,
            stride=1,
            padding=1,
            name="yolo_input")
        conv = self.downsample(
            conv1,
            ch_out=conv1.shape[1] * 2,
            name="yolo_input.downsample")
        
        for i, stage in enumerate(stages):
            conv = self.layer_warp(
                block_func,
                conv,
                32 * (2**i),
                stage,
                name="stage.{}".format(i))
            if i < len(stages) - 1:  # do not downsaple in the last stage
                conv = self.downsample(
                    conv,
                    ch_out=conv.shape[1] * 2,
                    name="stage.{}.downsample".format(i))
        pool = fluid.layers.pool2d(
            input=conv, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(input=pool,
                              size=class_dim,
                              param_attr=ParamAttr(
                              initializer=fluid.initializer.Uniform(-stdv, stdv),name='fc_weights'),
                              bias_attr=ParamAttr(name='fc_offset'))
        return out
        
        



    def conv_bn_layer(self, input, ch_out, filter_size, stride, padding, name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(name=name + ".conv.weights"),
            bias_attr=False)

        bn_name = name + ".bn"
        out = fluid.layers.batch_norm(
            input=conv,
            act='relu',
            param_attr=ParamAttr(name=bn_name + '.scale'),
            bias_attr=ParamAttr(name=bn_name + '.offset'),
            moving_mean_name=bn_name + '.mean',
            moving_variance_name=bn_name + '.var')
        return out



    def downsample(self, input, ch_out, filter_size=3, stride=2, padding=1, name=None):
        return self.conv_bn_layer(
            input,
            ch_out=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            name=name)


    def basicblock(self, input, ch_out, name=None):
        conv1 = self.conv_bn_layer(
            input, ch_out, 1, 1, 0, name=name + ".0")
        conv2 = self.conv_bn_layer(
            conv1, ch_out * 2, 3, 1, 1, name=name + ".1")
        out = fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        return out


    def layer_warp(self, block_func, input, ch_out, count, name=None):
        res_out = block_func(
            input, ch_out, name='{}.0'.format(name))
        for j in range(1, count):
            res_out = block_func(
                res_out, ch_out, name='{}.{}'.format(name, j))
        return res_out


