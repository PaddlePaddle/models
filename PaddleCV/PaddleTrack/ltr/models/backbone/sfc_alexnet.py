from collections import OrderedDict

from paddle import fluid
from paddle.fluid.dygraph import nn


class SFC_AlexNet(fluid.dygraph.Layer):
    def __init__(self, name, is_test):
        super(SFC_AlexNet, self).__init__()

        self.is_test = is_test
        self.layer_init()

    def layer_init(self):
        # for conv1
        self.conv1 = nn.Conv2D(
            num_channels=3,
            num_filters=96,
            filter_size=11,
            stride=2,
            padding=0,
            groups=1,
            param_attr=self.weight_init(),
            bias_attr=self.bias_init())
        self.bn1 = nn.BatchNorm(
            num_channels=96,
            is_test=self.is_test,
            param_attr=self.norm_weight_init(),
            bias_attr=self.bias_init(),
            use_global_stats=self.is_test)
        self.pool1 = nn.Pool2D(
            pool_size=3, pool_type="max", pool_stride=2, pool_padding=0)
        # for conv2
        self.conv2 = nn.Conv2D(
            num_channels=96,
            num_filters=256,
            filter_size=5,
            stride=1,
            padding=0,
            groups=2,
            param_attr=self.weight_init(),
            bias_attr=self.bias_init())
        self.bn2 = nn.BatchNorm(
            num_channels=256,
            is_test=self.is_test,
            param_attr=self.norm_weight_init(),
            bias_attr=self.bias_init(),
            use_global_stats=self.is_test)
        self.pool2 = nn.Pool2D(
            pool_size=3, pool_type="max", pool_stride=2, pool_padding=0)
        # for conv3
        self.conv3 = nn.Conv2D(
            num_channels=256,
            num_filters=384,
            filter_size=3,
            stride=1,
            padding=0,
            groups=1,
            param_attr=self.weight_init(),
            bias_attr=self.bias_init())
        self.bn3 = nn.BatchNorm(
            num_channels=384,
            is_test=self.is_test,
            param_attr=self.norm_weight_init(),
            bias_attr=self.bias_init(),
            use_global_stats=self.is_test)
        # for conv4
        self.conv4 = nn.Conv2D(
            num_channels=384,
            num_filters=384,
            filter_size=3,
            stride=1,
            padding=0,
            groups=2,
            param_attr=self.weight_init(),
            bias_attr=self.bias_init())
        self.bn4 = nn.BatchNorm(
            num_channels=384,
            is_test=self.is_test,
            param_attr=self.norm_weight_init(),
            bias_attr=self.bias_init(),
            use_global_stats=self.is_test)
        # for conv5
        self.conv5 = nn.Conv2D(
            num_channels=384,
            num_filters=256,
            filter_size=3,
            stride=1,
            padding=0,
            groups=2,
            param_attr=self.weight_init(),
            bias_attr=self.bias_init())

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, inputs, output_layers):
        outputs = OrderedDict()

        out1 = self.conv1(inputs)
        out1 = self.bn1(out1)
        out1 = fluid.layers.relu(out1)
        if self._add_output_and_check('conv1', out1, outputs, output_layers):
            return outputs

        out1 = self.pool1(out1)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = fluid.layers.relu(out2)
        if self._add_output_and_check('conv2', out2, outputs, output_layers):
            return outputs

        out2 = self.pool2(out2)

        out3 = self.conv3(out2)
        out3 = self.bn3(out3)
        out3 = fluid.layers.relu(out3)
        if self._add_output_and_check('conv3', out3, outputs, output_layers):
            return outputs

        out4 = self.conv4(out3)
        out4 = self.bn4(out4)
        out4 = fluid.layers.relu(out4)
        if self._add_output_and_check('conv4', out4, outputs, output_layers):
            return outputs

        out5 = self.conv5(out4)
        if self._add_output_and_check('conv5', out5, outputs, output_layers):
            return outputs

        return outputs

    def norm_weight_init(self):
        init = fluid.initializer.ConstantInitializer(1.0)
        param = fluid.ParamAttr(initializer=init)
        return param

    def weight_init(self):
        init = fluid.initializer.MSRAInitializer(uniform=False)
        param = fluid.ParamAttr(initializer=init)
        return param

    def bias_init(self):
        init = fluid.initializer.ConstantInitializer(value=0.)
        param = fluid.ParamAttr(initializer=init)
        return param
