"""
动态图构建 AlexNet
"""
import paddle.fluid as fluid
import numpy as np

class Conv2D(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):

        super(Conv2D, self).__init__(name_scope)

        self._conv2d = fluid.dygraph.Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            param_attr=param_attr,
            bias_attr=bias_attr,
            act=act,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        return x

class Conv2DPool(fluid.dygraph.Layer):
    def __init__(self,
                name_scope,
                num_channels,
                num_filters,
                filter_size,
                pool_size,
                pool_stride,
                pool_padding=0,
                pool_type='max',
                global_pooling=False,
                conv_stride=1,
                conv_padding=0,
                conv_dilation=1,
                conv_groups=1,
                act=None,
                use_cudnn=False,
                param_attr=None,
                bias_attr=None):

        super(Conv2DPool, self).__init__(name_scope)

        self._conv2d = fluid.dygraph.Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=param_attr,
            bias_attr=bias_attr,
            act=act,
            use_cudnn=use_cudnn)

        self._pool2d = fluid.dygraph.Pool2D(
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x

class AlexNet(fluid.dygraph.Layer):
    def __init__(self, name_scope, class_dim):
        super(AlexNet, self).__init__(name_scope)

        self.conv_pool_1 = Conv2DPool(self.full_name(), 3, 64, 11, 3, 2, conv_stride=4, conv_padding=2, act='relu')
        self.conv_pool_2 = Conv2DPool(self.full_name(), 64, 192, 5, 3, 2, conv_stride=1, conv_padding=2, act='relu')
        self.conv_3 = Conv2D(self.full_name(), 192, 384, 3, 1, 1, act='relu')
        self.conv_4 = Conv2D(self.full_name(), 384, 256, 3, 1, 1, act='relu')
        self.conv_pool_5 = Conv2DPool(self.full_name(), 256, 256, 3, 3, 2, conv_stride=1, conv_padding=1, act='relu')
        self.fc6 = fluid.dygraph.FC(self.full_name(), 9216, 4096, act='relu')
        self.fc7 = fluid.dygraph.FC(self.full_name(), 4096, 4096, act='relu')
        self.fc8 = fluid.dygraph.FC(self.full_name(), 4096, class_dim, act='softmax')

    def forward(self, inputs, label=None):
        out = self.conv_pool_1(inputs)
        out = self.conv_pool_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_pool_5(out)
        out = self.fc6(out)
        out = fluid.layers.dropout(out, 0.5)
        out = self.fc7(out)
        out = fluid.layers.dropout(out, 0.5)
        out = self.fc8(out)

        if label is not None:
            acc = fluid.layers.accuracy(input=out, label=label)
            return out, acc
        else:
            return out

if __name__ == '__main__':
    with fluid.dygraph.guard():
        alexnet = AlexNet('alex-net', 3)
        img = np.zeros([2, 3, 224, 224]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        outs = alexnet(img).numpy()

        print(outs)


