from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid
import math

#paper https://arxiv.org/abs/1801.04381
__all__ = ["MobileNetV2"]

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

class MobileNetV2():
    def __init__(self):
        self.params = train_parameters
        self.linear_bottleneck_setting = [
            # according to paper
            # t(expansion), c(output_channels), n(repeat_times), s(stride)
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
    def net(self, input, class_dim=100):
        input = self.conv_bn_layer(
            input,
            filter_size=3,
            num_filters=32,
            stride=2,
            padding=1,
            act='relu'
        )
        for t,c,n,s in self.linear_bottleneck_setting:
            for i in range(n):
                s=2 if s==2 and i==0 else 1
                print("s++++++++++++++++++",s)
                input = self.linear_bottleneck(
                                         input=input,
                                         output_channels = c,
                                         expansion = t,
                                         stride =s)
            print("====================")
        #conv2d 1x1 output_channels=1280 stride =1
        input = self.conv_bn_layer(
            input=input,
            filter_size=1,
            num_filters=1280,
            stride=1,
            padding=1,
            act='relu'
        )
        #global pool
        pool = fluid.layers.pool2d(
            input=input, pool_size=7, pool_type='avg', global_pooling=True)
        #con2d 1x1 Q:FC or conv2d? dropout?
        input = fluid.layers.dropout(x=pool,dropout_prob=0.2)
        out = fluid.layers.fc(input,size=class_dim,act='softmax')
        """
        out = fluid.layers.conv2d(
            input,
            filter_size=1,num_filters
            num_filters=class_dim
        )
        """
        return out
    # batch norm conv NCHW
    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      num_groups=1,
                      padding=0,
                      act=None):
        print("=========",input.shape[3],"X",input.shape[1],num_filters,stride,num_groups,padding)
        conv = fluid.layers.conv2d(
            input = input,
            num_filters = num_filters,
            filter_size=filter_size,
            stride=stride,
            groups=num_groups,
            padding=padding,
            act=None,
            bias_attr=False)

        return fluid.layers.batch_norm(input=conv,act=act)
#example input, ex:1 , nf:32 , s1
#conv0 input nf:32 , fs:1x1 ,channel = 32
#conv1 input nf:32 , fs:3x3 ,channel = 32/32=1
#conv2 input nf:16, fs:1x1 , channel=32


#only first time loop can use stride = 2
    def linear_bottleneck(self,
                         input,
                         output_channels,
                         expansion,
                         stride):
        assert stride in [1,2]
        
        input_channels = input.shape[1]
        conv_0 = self.conv_bn_layer(input=input,
                                    num_filters=input_channels*expansion,
                                    filter_size=1,
                                    stride=1,
                                    padding=0,
                                    num_groups=1,
                                    act='relu')
        conv_1 = self.conv_bn_layer(input=conv_0,
                                    num_filters=input_channels*expansion,
                                    filter_size=3,
                                    stride=stride,
                                    padding=1,
                                    num_groups=input_channels*expansion,
                                    act='relu')
        # mobilenetv2 does not apply relu when compose
        conv_2 = self.conv_bn_layer(input=conv_1,
                                    num_filters=output_channels,
                                    filter_size=1,
                                    stride=1,
                                    padding=0,
                                    num_groups=1)

# if stride = 2 does not shortcut
# if stride = 1 shortcut
# Q: elementwise_add use relu or not? A: linear bottleneck without relu
        if (stride == 1 and input_channels <> output_channels) or stride == 2:
            return conv_2
        if (stride==1 and input_channels==output_channels):
            return fluid.layers.elementwise_add(x=input, y=conv_2, act=None)

#In the paper, it have 19 layers, but actually, it has 17 layers

