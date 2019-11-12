# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import shutil
import paddle.fluid as fluid
import os


__all__ = ['DANet']


class ConvBN(fluid.dygraph.Layer):

    def __init__(self,
                 name_scope,
                 num_filters,
                 filter_size=3,
                 stride=1,
                 dilation=1,
                 act=None,
                 learning_rate=1.0,
                 dtype='float32',
                 bias_attr=False):
        super(ConvBN, self).__init__(name_scope)

        if dilation != 1:
            padding = dilation
        else:
            padding = (filter_size - 1) // 2

        self._conv = fluid.dygraph.Conv2D(name_scope,
                                          num_filters=num_filters,
                                          filter_size=filter_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          act=None,
                                          dtype=dtype,
                                          bias_attr=bias_attr if bias_attr is False else fluid.ParamAttr(
                                              learning_rate=learning_rate,
                                              name='bias'),
                                          param_attr=fluid.ParamAttr(
                                              learning_rate=learning_rate,
                                              name='weight')
                                          )
        self._bn = fluid.dygraph.BatchNorm(name_scope,
                                           num_channels=num_filters,
                                           act=act,
                                           dtype=dtype,
                                           momentum=0.9,
                                           epsilon=1e-5,
                                           bias_attr=fluid.ParamAttr(
                                               learning_rate=learning_rate,
                                               name='bias'),
                                           param_attr=fluid.ParamAttr(
                                               learning_rate=learning_rate,
                                               name='weight'),
                                           moving_mean_name='running_mean',
                                           moving_variance_name='running_var'
                                           )

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._bn(x)
        return x


class BasicBlock(fluid.dygraph.Layer):

    def __init__(self,
                 name_scope,
                 num_filters,
                 stride=1,
                 dilation=1,
                 same=False):
        super(BasicBlock, self).__init__(name_scope)
        self._conv0 = ConvBN(self.full_name(),
                             num_filters=num_filters,
                             filter_size=3,
                             stride=stride,
                             dilation=dilation,
                             act='relu')
        self._conv1 = ConvBN(self.full_name(),
                             num_filters=num_filters,
                             filter_size=3,
                             stride=1,
                             dilation=dilation,
                             act=None)

        self.same = same

        if not same:
            self._skip = ConvBN(self.full_name(),
                                num_filters=num_filters,
                                filter_size=1,
                                stride=stride,
                                act=None)

    def forward(self, inputs):
        x = self._conv0(inputs)
        x = self._conv1(x)
        if self.same:
            skip = inputs
        else:
            skip = self._skip(inputs)
        x = fluid.layers.elementwise_add(x, skip, act='relu')
        return x


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_filters, stride, dilation=1, same=False):
        super(BottleneckBlock, self).__init__(name_scope)
        self.expansion = 4

        self._conv0 = ConvBN(name_scope,
                             num_filters=num_filters,
                             filter_size=1,
                             stride=1,
                             act='relu')
        self._conv1 = ConvBN(name_scope,
                             num_filters=num_filters,
                             filter_size=3,
                             stride=stride,
                             dilation=dilation,
                             act='relu')
        self._conv2 = ConvBN(name_scope,
                             num_filters=num_filters * self.expansion,
                             filter_size=1,
                             stride=1,
                             act=None)
        self.same = same

        if not same:
            self._skip = ConvBN(name_scope,
                                num_filters=num_filters * self.expansion,
                                filter_size=1,
                                stride=stride,
                                act=None)

    def forward(self, inputs):
        x = self._conv0(inputs) 
        x = self._conv1(x)  
        x = self._conv2(x) 
        if self.same:
            skip = inputs
        else:
            skip = self._skip(inputs)
        x = fluid.layers.elementwise_add(x, skip, act='relu')
        return x


class ResNet(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 layer=152,
                 num_class=1000,
                 dilated=True,
                 multi_grid=True,
                 multi_dilation=[4, 8, 16],
                 need_fc=False):
        super(ResNet, self).__init__(name_scope)

        support_layer = [18, 34, 50, 101, 152]
        assert layer in support_layer, 'layer({}) not in {}'.format(layer, support_layer)
        self.need_fc = need_fc
        self.num_filters_list = [64, 128, 256, 512]
        if layer == 18:
            self.depth = [2, 2, 2, 2]
        elif layer == 34:
            self.depth = [3, 4, 6, 3]
        elif layer == 50:
            self.depth = [3, 4, 6, 3]
        elif layer == 101:
            self.depth = [3, 4, 23, 3]
        elif layer == 152:
            self.depth = [3, 8, 36, 3]

        if multi_grid:
            assert multi_dilation is not None
            self.multi_dilation = multi_dilation

        self._conv = ConvBN(name_scope, 64, 7, 2, act='relu')
        self._pool = fluid.dygraph.Pool2D(name_scope,
                                          pool_size=3,
                                          pool_stride=2,
                                          pool_padding=1,
                                          pool_type='max')
        if layer >= 50:
            self.layer1 = self._make_layer(block=BottleneckBlock,
                                           depth=self.depth[0],
                                           num_filters=self.num_filters_list[0],
                                           stride=1,
                                           same=False,
                                           name='layer1')
            self.layer2 = self._make_layer(block=BottleneckBlock,
                                           depth=self.depth[1],
                                           num_filters=self.num_filters_list[1],
                                           stride=2,
                                           same=False,
                                           name='layer2')
            if dilated:
                self.layer3 = self._make_layer(block=BottleneckBlock,
                                               depth=self.depth[2],
                                               num_filters=self.num_filters_list[2],
                                               stride=2,
                                               dilation=2,
                                               same=False,
                                               name='layer3')
                if multi_grid:  # layer4 采用不同的采样率
                    self.layer4 = self._make_layer(block=BottleneckBlock,
                                                   depth=self.depth[3],
                                                   num_filters=self.num_filters_list[3],
                                                   stride=2,
                                                   dilation=4,
                                                   multi_grid=multi_grid,
                                                   multi_dilation=self.multi_dilation,
                                                   same=False,
                                                   name='layer4')
                else:
                    self.layer4 = self._make_layer(block=BottleneckBlock,
                                                   depth=self.depth[3],
                                                   num_filters=self.num_filters_list[3],
                                                   stride=2,
                                                   dilation=4,
                                                   same=False,
                                                   name='layer4')
            else:
                self.layer3 = self._make_layer(block=BottleneckBlock,
                                               depth=self.depth[2],
                                               num_filters=self.num_filters_list[2],
                                               stride=2,
                                               dilation=1,
                                               same=False,
                                               name='layer3')
                self.layer4 = self._make_layer(block=BottleneckBlock,
                                               depth=self.depth[3],
                                               num_filters=self.num_filters_list[3],
                                               stride=2,
                                               dilation=1,
                                               same=False,
                                               name='layer4')

        else:  # layer=18 or layer=34
            self.layer1 = self._make_layer(block=BasicBlock,
                                           depth=self.depth[0],
                                           num_filters=self.num_filters_list[0],
                                           stride=1,
                                           same=True,
                                           name=name_scope)
            self.layer2 = self._make_layer(block=BasicBlock,
                                           depth=self.depth[1],
                                           num_filters=self.num_filters_list[1],
                                           stride=2,
                                           same=False,
                                           name=name_scope)
            self.layer3 = self._make_layer(block=BasicBlock,
                                           depth=self.depth[2],
                                           num_filters=self.num_filters_list[2],
                                           stride=2,
                                           dilation=1,
                                           same=False,
                                           name=name_scope)
            self.layer4 = self._make_layer(block=BasicBlock,
                                           depth=self.depth[3],
                                           num_filters=self.num_filters_list[3],
                                           stride=2,
                                           dilation=1,
                                           same=False,
                                           name=name_scope)

        self._avgpool = fluid.dygraph.Pool2D(name_scope,
                                             global_pooling=True,
                                             pool_type='avg')
        self.fc = fluid.dygraph.FC(name_scope,
                                   size=num_class,
                                   act='softmax')

    def _make_layer(self, block, depth, num_filters, stride=1, dilation=1, same=False, multi_grid=False,
                    multi_dilation=None, name=None):
        layers = []
        if dilation != 1:
            #  stride(2x2) with a dilated convolution instead
            stride = 1

        if multi_grid:
            assert len(multi_dilation) == 3
            for depth in range(depth):
                temp = block(name + '.{}'.format(depth),
                             num_filters=num_filters,
                             stride=stride,
                             dilation=multi_dilation[depth],
                             same=same)
                stride = 1
                same = True
                layers.append(self.add_sublayer('_{}_{}'.format(name, depth + 1), temp))
        else:
            for depth in range(depth):
                temp = block(name + '.{}'.format(depth),
                             num_filters=num_filters,
                             stride=stride,
                             dilation=dilation if depth > 0 else 1,
                             same=same)
                stride = 1
                same = True
                layers.append(self.add_sublayer('_{}_{}'.format(name, depth + 1), temp))
        return layers

    def forward(self, inputs):
        x = self._conv(inputs)

        x = self._pool(x)
        for layer in self.layer1:
            x = layer(x)
        c1 = x

        for layer in self.layer2:
            x = layer(x)
        c2 = x

        for layer in self.layer3:
            x = layer(x)
        c3 = x

        for layer in self.layer4:
            x = layer(x)
        c4 = x

        if self.need_fc:
            x = self._avgpool(x)
            x = self.fc(x)
            return x
        else:  
            return c1, c2, c3, c4


class CAM(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 in_channels=512,
                 default_value=0):
        """
        channel_attention_module
        """
        super(CAM, self).__init__(name_scope)
        self.in_channels = in_channels
        self.gamma = fluid.layers.create_parameter(shape=[1],
                                                   dtype='float32',
                                                   is_bias=True,
                                                   attr=fluid.ParamAttr(
                                                       learning_rate=10.0,
                                                       name='cam_gamma'),
                                                   default_initializer=fluid.initializer.ConstantInitializer(
                                                       value=default_value)
                                                   )

    def forward(self, inputs):
        batch_size, c, h, w = inputs.shape
        out_b = fluid.layers.reshape(inputs, shape=[batch_size, self.in_channels, h * w])  
        out_c = fluid.layers.reshape(inputs, shape=[batch_size, self.in_channels, h * w])  
        out_c_t = fluid.layers.transpose(out_c, perm=[0, 2, 1])  
        mul_bc = fluid.layers.matmul(out_b, out_c_t)  

        mul_bc_max = fluid.layers.reduce_max(mul_bc, dim=-1, keep_dim=True)
        mul_bc_max = fluid.layers.expand(mul_bc_max, expand_times=[1, 1, c])
        x = fluid.layers.elementwise_sub(mul_bc_max, mul_bc)  

        attention = fluid.layers.softmax(x, use_cudnn=True, axis=-1)  

        out_d = fluid.layers.reshape(inputs, shape=[batch_size, self.in_channels, h * w])  
        attention_mul = fluid.layers.matmul(attention, out_d)  

        attention_reshape = fluid.layers.reshape(attention_mul, shape=[batch_size, self.in_channels, h, w])  
        gamma_attention = fluid.layers.elementwise_mul(attention_reshape, self.gamma)  
        out = fluid.layers.elementwise_add(gamma_attention, inputs)  
        return out


class PAM(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 in_channels=512,
                 default_value=0):
        """
        position_attention_module
        """
        super(PAM, self).__init__(name_scope)

        assert in_channels // 8, 'in_channel // 8 > 0 '
        self.channel_in = in_channels // 8
        self._convB = fluid.dygraph.Conv2D(name_scope,
                                           num_filters=in_channels // 8,
                                           filter_size=1,
                                           bias_attr=fluid.ParamAttr(
                                               learning_rate=10.0,
                                               name='bias'),
                                           param_attr=fluid.ParamAttr(
                                               learning_rate=10.0,
                                               name='weight')
                                           )
        self._convC = fluid.dygraph.Conv2D(name_scope,
                                           num_filters=in_channels // 8,
                                           filter_size=1,
                                           bias_attr=fluid.ParamAttr(
                                               learning_rate=10.0,
                                               name='bias'),
                                           param_attr=fluid.ParamAttr(
                                               learning_rate=10.0,
                                               name='weight')
                                           )
        self._convD = fluid.dygraph.Conv2D(name_scope,
                                           num_filters=in_channels,
                                           filter_size=1,
                                           bias_attr=fluid.ParamAttr(
                                               learning_rate=10.0,
                                               name='bias'),
                                           param_attr=fluid.ParamAttr(
                                               learning_rate=10.0,
                                               name='weight')
                                           )
        self.gamma = fluid.layers.create_parameter(shape=[1],
                                                   dtype='float32',
                                                   is_bias=True,
                                                   attr=fluid.ParamAttr(
                                                       learning_rate=10.0,
                                                       name='pam_gamma'),
                                                   default_initializer=fluid.initializer.ConstantInitializer(
                                                       value=default_value))

    def forward(self, inputs):
        batch_size, c, h, w = inputs.shape
        out_b = self._convB(inputs)  
        out_b_reshape = fluid.layers.reshape(out_b, shape=[batch_size, self.channel_in, h * w])  
        out_b_reshape_t = fluid.layers.transpose(out_b_reshape, perm=[0, 2, 1])  
        out_c = self._convC(inputs)  
        out_c_reshape = fluid.layers.reshape(out_c, shape=[batch_size, self.channel_in, h * w])  

        mul_bc = fluid.layers.matmul(out_b_reshape_t, out_c_reshape)  
        soft_max_bc = fluid.layers.softmax(mul_bc, use_cudnn=True, axis=-1)  

        out_d = self._convD(inputs)  
        out_d_reshape = fluid.layers.reshape(out_d, shape=[batch_size, self.channel_in * 8, h * w])  
        attention = fluid.layers.matmul(out_d_reshape, fluid.layers.transpose(soft_max_bc, perm=[0, 2, 1]))
        attention = fluid.layers.reshape(attention, shape=[batch_size, self.channel_in * 8, h, w])  

        gamma_attention = fluid.layers.elementwise_mul(attention, self.gamma)  
        out = fluid.layers.elementwise_add(gamma_attention, inputs)  
        return out


class DAHead(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 in_channels,
                 out_channels,
                 batch_size):
        super(DAHead, self).__init__(name_scope)
        self.in_channel = in_channels // 4
        self.batch_size = batch_size
        self._conv_bn_relu0 = ConvBN(name_scope,
                                     num_filters=self.in_channel,
                                     filter_size=3,
                                     stride=1,
                                     act='relu',
                                     learning_rate=10.0,
                                     bias_attr=False)

        self._conv_bn_relu1 = ConvBN(name_scope,
                                     num_filters=self.in_channel,
                                     filter_size=3,
                                     stride=1,
                                     act='relu',
                                     learning_rate=10.0,
                                     bias_attr=False)

        self._pam = PAM('pam', in_channels=self.in_channel, default_value=0.0)
        self._cam = CAM('cam', in_channels=self.in_channel, default_value=0.0)

        self._conv_bn_relu2 = ConvBN(name_scope,
                                     num_filters=self.in_channel,
                                     filter_size=3,
                                     stride=1,
                                     act='relu',
                                     learning_rate=10.0,
                                     bias_attr=False)

        self._conv_bn_relu3 = ConvBN(name_scope,
                                     num_filters=self.in_channel,
                                     filter_size=3,
                                     stride=1,
                                     act='relu',
                                     learning_rate=10.0,
                                     bias_attr=False)
        self._pam_last_conv = fluid.dygraph.Conv2D(name_scope,
                                                   num_filters=out_channels,
                                                   filter_size=1,
                                                   bias_attr=fluid.ParamAttr(
                                                       learning_rate=10.0,
                                                       name='bias'),
                                                   param_attr=fluid.ParamAttr(
                                                       learning_rate=10.0,
                                                       name='weight')
                                                   )
        self._cam_last_conv = fluid.dygraph.Conv2D(name_scope,
                                                   num_filters=out_channels,
                                                   filter_size=1,
                                                   bias_attr=fluid.ParamAttr(
                                                       learning_rate=10.0,
                                                       name='bias'),
                                                   param_attr=fluid.ParamAttr(
                                                       learning_rate=10.0,
                                                       name='weight')
                                                   )
        self._last_conv = fluid.dygraph.Conv2D(name_scope,
                                               num_filters=out_channels,
                                               filter_size=1,
                                               bias_attr=fluid.ParamAttr(
                                                   learning_rate=10.0,
                                                   name='bias'),
                                               param_attr=fluid.ParamAttr(
                                                   learning_rate=10.0,
                                                   name='weight')
                                               )

    def forward(self, inputs):
        out = []
        inputs_pam = self._conv_bn_relu0(inputs)
        pam = self._pam(inputs_pam)
        position = self._conv_bn_relu2(pam)

        batch_size, num_channels = position.shape[:2]

        # dropout2d
        ones = fluid.layers.ones(shape=[self.batch_size, num_channels], dtype='float32')
        dropout1d_P = fluid.layers.dropout(ones, 0.1)
        out_position_drop2d = fluid.layers.elementwise_mul(position, dropout1d_P, axis=0)
        dropout1d_P.stop_gradient = True

        inputs_cam = self._conv_bn_relu1(inputs)
        cam = self._cam(inputs_cam)
        channel = self._conv_bn_relu3(cam)

        # dropout2d
        ones2 = fluid.layers.ones(shape=[self.batch_size, num_channels], dtype='float32')
        dropout1d_C = fluid.layers.dropout(ones2, 0.1)
        out_channel_drop2d = fluid.layers.elementwise_mul(channel, dropout1d_C, axis=0)
        dropout1d_C.stop_gradient = True
        position_out = self._pam_last_conv(out_position_drop2d)
        channel_out = self._cam_last_conv(out_channel_drop2d)

        feat_sum = fluid.layers.elementwise_add(position, channel, axis=1)
        feat_sum_batch_size, feat_sum_num_channels = feat_sum.shape[:2]  

        # dropout2d
        feat_sum_ones = fluid.layers.ones(shape=[self.batch_size, feat_sum_num_channels], dtype='float32')
        dropout1d_sum = fluid.layers.dropout(feat_sum_ones, 0.1)
        dropout2d_feat_sum = fluid.layers.elementwise_mul(feat_sum, dropout1d_sum, axis=0)
        dropout1d_sum.stop_gradient = True
        feat_sum_out = self._last_conv(dropout2d_feat_sum)

        out.append(feat_sum_out)
        out.append(position_out)
        out.append(channel_out)
        return tuple(out)


class DANet(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 backbone='resnet50',
                 num_classes=19,
                 batch_size=1,
                 dilated=True,
                 multi_grid=True,
                 multi_dilation=[4, 8, 16]):
        super(DANet, self).__init__(name_scope)
        if backbone == 'resnet50':
            print('backbone resnet50, dilated={}, multi_grid={}, '
                  'multi_dilation={}'.format(dilated, multi_grid, multi_dilation))
            self._backone = ResNet('resnet50', layer=50, dilated=dilated,
                                   multi_grid=multi_grid, multi_dilation=multi_dilation)
        elif backbone == 'resnet101':
            print('backbone resnet101, dilated={}, multi_grid={}, '
                  'multi_dilation={}'.format(dilated, multi_grid, multi_dilation))
            self._backone = ResNet('resnet101', layer=101, dilated=dilated,
                                   multi_grid=multi_grid, multi_dilation=multi_dilation)
        elif backbone == 'resnet152':
            print('backbone resnet152, dilated={}, multi_grid={}, '
                  'multi_dilation={}'.format(dilated, multi_grid, multi_dilation))
            self._backone = ResNet('resnet152', layer=152, dilated=dilated,
                                   multi_grid=multi_grid, multi_dilation=multi_dilation)
        else:
            raise ValueError('unknown backbone: {}'.format(backbone))

        self._head = DAHead('DA_head', in_channels=2048, out_channels=num_classes, batch_size=batch_size)

    def forward(self, inputs):
        h, w = inputs.shape[2:]
        _, _, c3, c4 = self._backone(inputs)
        x1, x2, x3 = self._head(c4)
        out = []
        out1 = fluid.layers.resize_bilinear(x1, out_shape=[h, w])
        out2 = fluid.layers.resize_bilinear(x2, out_shape=[h, w])
        out3 = fluid.layers.resize_bilinear(x3, out_shape=[h, w])
        out.append(out1)
        out.append(out2)
        out.append(out3)
        return out


def copy_model(path, new_path):
    shutil.rmtree(new_path, ignore_errors=True)
    shutil.copytree(path, new_path)
    model_path = os.path.join(new_path, '__model__')
    if os.path.exists(model_path):
        os.remove(model_path)


if __name__ == '__main__':
    import numpy as np

    with fluid.dygraph.guard(fluid.CPUPlace()):
        x = np.random.randn(2, 3, 224, 224).astype('float32')
        x = fluid.dygraph.to_variable(x)
        model = DANet('test', backbone='resnet101', num_classes=19, batch_size=2)
        y = model(x)
        print(y[0].shape)
