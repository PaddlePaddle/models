#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from paddle.fluid.initializer import MSRA
import paddle.fluid as fluid


# get init parameters for conv layer
def get_conv_init(fan_out):
    return MSRA(uniform=False, fan_in=fan_out)


"""Video models."""


class BottleneckTransform(fluid.dygraph.Layer):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """

    def __init__(
            self,
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=False,
            inplace_relu=True,
            eps=1e-5,
            dilation=1, ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            dilation (int): size of dilation.
        """
        super(BottleneckTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._stride_1x1 = stride_1x1
        self._construct(dim_in, dim_out, stride, dim_inner, num_groups,
                        dilation)

    def _construct(self, dim_in, dim_out, stride, dim_inner, num_groups,
                   dilation):
        str1x1, str3x3 = (stride, 1) if self._stride_1x1 else (1, stride)

        fan = (dim_inner) * (self.temp_kernel_size * 1 * 1)
        initializer_tmp = get_conv_init(fan)
        batchnorm_weight = 1.0

        self.a = fluid.dygraph.nn.Conv3D(
            num_channels=dim_in,
            num_filters=dim_inner,
            filter_size=[self.temp_kernel_size, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[int(self.temp_kernel_size // 2), 0, 0],
            param_attr=fluid.ParamAttr(initializer=initializer_tmp),
            bias_attr=False)
        self.a_bn = fluid.dygraph.BatchNorm(
            num_channels=dim_inner,
            epsilon=self._eps,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(batchnorm_weight),
                regularizer=fluid.regularizer.L2Decay(
                    regularization_coeff=0.0)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.0),
                regularizer=fluid.regularizer.L2Decay(
                    regularization_coeff=0.0)))

        # 1x3x3, BN, ReLU.
        fan = (dim_inner) * (1 * 3 * 3)
        initializer_tmp = get_conv_init(fan)
        batchnorm_weight = 1.0

        self.b = fluid.dygraph.nn.Conv3D(
            num_channels=dim_inner,
            num_filters=dim_inner,
            filter_size=[1, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[0, dilation, dilation],
            groups=num_groups,
            dilation=[1, dilation, dilation],
            param_attr=fluid.ParamAttr(initializer=initializer_tmp),
            bias_attr=False)
        self.b_bn = fluid.dygraph.BatchNorm(
            num_channels=dim_inner,
            epsilon=self._eps,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(batchnorm_weight),
                regularizer=fluid.regularizer.L2Decay(
                    regularization_coeff=0.0)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.0),
                regularizer=fluid.regularizer.L2Decay(
                    regularization_coeff=0.0)))

        # 1x1x1, BN.
        fan = (dim_out) * (1 * 1 * 1)
        initializer_tmp = get_conv_init(fan)
        batchnorm_weight = 0.0

        self.c = fluid.dygraph.nn.Conv3D(
            num_channels=dim_inner,
            num_filters=dim_out,
            filter_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            param_attr=fluid.ParamAttr(initializer=initializer_tmp),
            bias_attr=False)
        self.c_bn = fluid.dygraph.BatchNorm(
            num_channels=dim_out,
            epsilon=self._eps,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(batchnorm_weight),
                regularizer=fluid.regularizer.L2Decay(
                    regularization_coeff=0.0)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.0),
                regularizer=fluid.regularizer.L2Decay(
                    regularization_coeff=0.0)))

    def forward(self, x):
        # Branch2a.
        x = self.a(x)
        x = self.a_bn(x)
        x = fluid.layers.relu(x)

        # Branch2b.
        x = self.b(x)
        x = self.b_bn(x)
        x = fluid.layers.relu(x)

        # Branch2c
        x = self.c(x)
        x = self.c_bn(x)
        return x


class ResBlock(fluid.dygraph.Layer):
    """
    Residual block.
    """

    def __init__(
            self,
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups=1,
            stride_1x1=False,
            inplace_relu=True,
            eps=1e-5,
            dilation=1, ):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            trans_func (string): transform function to be used to construct the
                bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            dilation (int): size of dilation.
        """
        super(ResBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._construct(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation, )

    def _construct(
            self,
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation, ):
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride != 1):
            fan = (dim_out) * (1 * 1 * 1)
            initializer_tmp = get_conv_init(fan)
            batchnorm_weight = 1.0
            self.branch1 = fluid.dygraph.nn.Conv3D(
                num_channels=dim_in,
                num_filters=dim_out,
                filter_size=1,
                stride=[1, stride, stride],
                padding=0,
                param_attr=fluid.ParamAttr(initializer=initializer_tmp),
                bias_attr=False,
                dilation=1)
            self.branch1_bn = fluid.dygraph.BatchNorm(
                num_channels=dim_out,
                epsilon=self._eps,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(batchnorm_weight),
                    regularizer=fluid.regularizer.L2Decay(
                        regularization_coeff=0.0)),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(0.0),
                    regularizer=fluid.regularizer.L2Decay(
                        regularization_coeff=0.0)))

        self.branch2 = BottleneckTransform(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=dilation, )

    def forward(self, x):
        if hasattr(self, "branch1"):
            x1 = self.branch1(x)
            x1 = self.branch1_bn(x1)
            x2 = self.branch2(x)
            x = fluid.layers.elementwise_add(x=x1, y=x2)
        else:
            x2 = self.branch2(x)
            x = fluid.layers.elementwise_add(x=x, y=x2)

        x = fluid.layers.relu(x)
        return x


class ResStage(fluid.dygraph.Layer):
    """
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        multi-pathway (SlowFast) cases.  More details can be found here:

        Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
        "Slowfast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(
            self,
            dim_in,
            dim_out,
            stride,
            temp_kernel_sizes,
            num_blocks,
            dim_inner,
            num_groups,
            num_block_temp_kernel,
            dilation,
            stride_1x1=False,
            inplace_relu=True, ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            temp_kernel_sizes (list): list of the p temporal kernel sizes of the
                convolution in the bottleneck. Different temp_kernel_sizes
                control different pathway.
            stride (list): list of the p strides of the bottleneck. Different
                stride control different pathway.
            num_blocks (list): list of p numbers of blocks for each of the
                pathway.
            dim_inner (list): list of the p inner channel dimensions of the
                input. Different channel dimensions control the input dimension
                of different pathways.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            num_block_temp_kernel (list): extent the temp_kernel_sizes to
                num_block_temp_kernel blocks, then fill temporal kernel size
                of 1 for the rest of the layers.
            dilation (list): size of dilation for each pathway.
        """
        super(ResStage, self).__init__()
        assert all((num_block_temp_kernel[i] <= num_blocks[i]
                    for i in range(len(temp_kernel_sizes))))
        self.num_blocks = num_blocks
        self.temp_kernel_sizes = [(temp_kernel_sizes[i] * num_blocks[i]
                                   )[:num_block_temp_kernel[i]] + [1] *
                                  (num_blocks[i] - num_block_temp_kernel[i])
                                  for i in range(len(temp_kernel_sizes))]
        assert (len({
            len(dim_in),
            len(dim_out),
            len(temp_kernel_sizes),
            len(stride),
            len(num_blocks),
            len(dim_inner),
            len(num_groups),
            len(num_block_temp_kernel),
        }) == 1)
        self.num_pathways = len(self.num_blocks)
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation, )

    def _construct(
            self,
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation, ):

        for pathway in range(self.num_pathways):
            for i in range(self.num_blocks[pathway]):
                res_block = ResBlock(
                    dim_in[pathway] if i == 0 else dim_out[pathway],
                    dim_out[pathway],
                    self.temp_kernel_sizes[pathway][i],
                    stride[pathway] if i == 0 else 1,
                    dim_inner[pathway],
                    num_groups[pathway],
                    stride_1x1=stride_1x1,
                    inplace_relu=inplace_relu,
                    dilation=dilation[pathway], )
                self.add_sublayer("pathway{}_res{}".format(pathway, i),
                                  res_block)

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]

            for i in range(self.num_blocks[pathway]):
                m = getattr(self, "pathway{}_res{}".format(pathway, i))
                x = m(x)
            output.append(x)

        return output


"""ResNe(X)t Head helper."""


class ResNetBasicHead(fluid.dygraph.Layer):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
            self,
            dim_in,
            num_classes,
            pool_size,
            dropout_rate=0.0, ):
        """
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
        """
        super(ResNetBasicHead, self).__init__()
        assert (len({len(pool_size), len(dim_in)}) == 1
                ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        fc_init_std = 0.01
        initializer_tmp = fluid.initializer.NormalInitializer(
            loc=0.0, scale=fc_init_std)
        self.projection = fluid.dygraph.Linear(
            input_dim=sum(dim_in),
            output_dim=num_classes,
            param_attr=fluid.ParamAttr(initializer=initializer_tmp),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.0)), )

    def forward(self, inputs, training):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            tmp_out = fluid.layers.pool3d(
                input=inputs[pathway],
                pool_type="avg",
                pool_size=self.pool_size[pathway],
                pool_stride=1,
                data_format="NCDHW")
            pool_out.append(tmp_out)

        x = fluid.layers.concat(input=pool_out, axis=1, name=None)
        x = fluid.layers.transpose(x=x, perm=(0, 2, 3, 4, 1))

        # Perform dropout.
        if self.dropout_rate > 0.0:
            x = fluid.layers.dropout(
                x,
                dropout_prob=self.dropout_rate,
                dropout_implementation='upscale_in_train')

        x = self.projection(x)

        # Performs fully convlutional inference.
        if not training:
            x = fluid.layers.softmax(x, axis=4)
            x = fluid.layers.reduce_mean(x, dim=[1, 2, 3])

        x = fluid.layers.reshape(x, shape=(x.shape[0], -1))
        return x
