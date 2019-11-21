#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import math
import numpy as np
from six.moves import reduce

from copy import deepcopy

import paddle
from paddle import fluid
import paddle.fluid.dygraph as dg
from paddle.fluid import core
from paddle.fluid.layers import utils
from paddle.fluid.framework import Variable
from paddle.fluid.initializer import Normal, Constant, NumpyArrayInitializer


def _norm(p, dim):
    """Computes the norm over all dimensions except dim.
    It differs from pytorch implementation that it does not keep dim.
    This difference is related with the broadcast mechanism in paddle.
    Read elementeise_mul for more.
    """

    if dim is None:
        return np.linalg.norm(p, ord=2, axis=None)
    elif dim == 0:
        p = np.reshape(p, newshape=(p.shape[0], -1))
        return np.linalg.norm(p, ord=2, axis=1)
    elif dim == p.ndim - 1:
        p = np.reshape(p, newshape=(-1, p.shape[-1]))
        return np.linalg.norm(p, ord=2, axis=0)
    else:
        perm = list(range(p.ndim))
        perm[0] = dim
        perm[dim] = 0
        return _norm(np.transpose(p, axes=perm))


class FC(dg.Layer):
    """
    **Fully Connected Layer**

    This function creates a fully connected layer in the network. It can take
    one or multiple tensors as its inputs(input can be a list of Variable, see
    Args in detail). It creates a pair of variables called (magnitude(g), 
    direction(V)) for each input tensor. Elementwise_mul(V, g) represents a fully connected 
    weight matrix from each input unit to each output unit. 
    The fully connected layer multiplies each input tensor
    with its corresponding weight to produce an output Tensor with shape [M, `size`],
    where M is batch size. If multiple input tensors are given, the results of
    multiple output tensors with shape [M, `size`] will be summed up. If bias_attr
    is not None, a bias variable will be created and added to the output.
    Finally, if activation is not None, it will be applied to the output as well.

    When the input is single tensor:

    .. math::

        Out = Act({X(normalize(V)g) + b})

    When the input are multiple tensors:

    .. math::

        Out = Act({\sum_{i=0}^{N-1}X_i(V_ig_i) + b})

    In the above equation:

    * :math:`N`: Number of the input. N equals to len(input) if input is list of Variable.
    * :math:`X_i`: The i-th input tensor.
    * :math:`V_i`: The i-th direction matrix corresponding i-th input tensor.
    * :math:`g_i`: The i-th magnitude vector corresponding i-th input tensor.
    * :math:`b`: The bias parameter created by this layer (if needed).
    * :math:`Act`: The activation function.
    * :math:`Out`: The output tensor.

    See below for an example.

    .. code-block:: text

        Given:
            data_1.data = [[[0.1, 0.2],
                           [0.3, 0.4]]]
            data_1.shape = (1, 2, 2) # 1 is batch_size

            data_2 = [[[0.1, 0.2, 0.3]]]
            data_2.shape = (1, 1, 3)

            out = fluid.layers.fc(input=[data_1, data_2], size=2)

        Then:
            out.data = [[0.18669507, 0.1893476]]
            out.shape = (1, 2)

    Args:
        name_scope(str): The name of this class.
        size(int): The number of output units in this layer.
        num_flatten_dims (int): The fc layer can accept an input tensor with more than
            two dimensions. If this happens, the multidimensional tensor will first be flattened
            into a 2-dimensional matrix. The parameter `num_flatten_dims` determines how the input
            tensor is flattened: the first `num_flatten_dims` (inclusive, index starts from 1)
            dimensions will be flatten to form the first dimension of the final matrix (height of
            the matrix), and the rest `rank(X) - num_flatten_dims` dimensions are flattened to
            form the second dimension of the final matrix (width of the matrix). For example, suppose
            `X` is a 5-dimensional tensor with a shape [2, 3, 4, 5, 6], and `num_flatten_dims` = 3.
            Then, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30]. Default: 1
        param_attr (ParamAttr|list of ParamAttr|None): The parameter attribute for learnable
            parameters/weights of this layer.
        bias_attr (ParamAttr|list of ParamAttr, default None): The parameter attribute for the bias
            of this layer. If it is set to False, no bias will be added to the output units.
            If it is set to None, the bias is initialized zero. Default: None.
        act (str|None): Activation to be applied to the output of this layer.
        is_test(bool): A flag indicating whether execution is in test phase. Default: False
        dtype(str): Dtype used for weight

    Raises:
        ValueError: If rank of the input tensor is less than 2.

    Examples:
        .. code-block:: python

          from paddle.fluid.dygraph.base import to_variable
          import paddle.fluid as fluid
          from paddle.fluid.dygraph import FC
          import numpy as np

          data = np.random.uniform( -1, 1, [30, 10, 32] ).astype('float32')
          with fluid.dygraph.guard():
              fc = FC( "fc", 64, num_flatten_dims=2)
              data = to_variable( data )
              conv = fc( data )

    """

    def __init__(self,
                 name_scope,
                 size,
                 num_flatten_dims=1,
                 epsilon=1e-30,
                 param_attr=None,
                 bias_attr=None,
                 act=None,
                 is_test=False,
                 dtype="float32"):
        super(FC, self).__init__(name_scope, dtype)

        self._size = size
        self._num_flatten_dims = num_flatten_dims
        self._epsilon = epsilon
        self._dtype = dtype
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act
        self.__g = list()
        self.__v = list()

    @property
    def _v(self, i=0):
        return self.__v[i]

    @property
    def _g(self, i=0):
        return self.__g[i]

    @_v.setter
    def _v(self, value, i=0):
        assert isinstance(value, Parameter)
        self.__v[i] = value

    @_g.setter
    def _g(self, value, i=0):
        assert isinstance(value, Parameter)
        self.__g[i] = value

    def _build_once(self, input):
        i = 0
        for inp, param in self._helper.iter_inputs_and_params(input,
                                                              self._param_attr):
            input_shape = inp.shape

            param_shape = [
                reduce(lambda a, b: a * b, input_shape[self._num_flatten_dims:],
                       1)
            ] + [self._size]
            self.__v.append(
                self.add_parameter(
                    "_v%d" % i,
                    self.create_parameter(
                        attr=param,
                        shape=param_shape,
                        dtype=self._dtype,
                        is_bias=False)))

            magnitude_shape = param_shape[1:]
            magnitude_value = np.linalg.norm(self.__v[i].numpy(), ord=2, axis=0)

            self.__g.append(
                self.add_parameter(
                    "_g%d" % i,
                    self.create_parameter(
                        attr=fluid.ParamAttr(
                            initializer=fluid.initializer.NumpyArrayInitializer(
                                magnitude_value)),
                        shape=magnitude_shape,
                        dtype=self._dtype,
                        is_bias=False)))
            i += 1

        size = list([self._size])
        self._b = self.create_parameter(
            attr=self._bias_attr, shape=size, dtype=self._dtype, is_bias=True)

    def forward(self, input):
        mul_results = list()
        i = 0
        for inp, param in self._helper.iter_inputs_and_params(input,
                                                              self._param_attr):
            v_norm = self._helper.create_variable_for_type_inference(
                self._dtype)
            v_normalized = self._helper.create_variable_for_type_inference(
                self._dtype)
            self._helper.append_op(
                type="norm",
                inputs={"X": self.__v[i]},
                outputs={"Out": v_normalized,
                         "Norm": v_norm},
                attrs={"axis": 0,
                       "epsilon": self._epsilon})
            weight = self._helper.create_variable_for_type_inference(
                self._dtype)
            self._helper.append_op(
                type="elementwise_mul",
                inputs={"X": [v_normalized],
                        "Y": [self.__g[i]]},
                outputs={"Out": [weight]},
                attrs={"axis": 1})
            tmp = self._helper.create_variable_for_type_inference(self._dtype)
            self._helper.append_op(
                type="mul",
                inputs={"X": inp,
                        "Y": weight},
                outputs={"Out": tmp},
                attrs={
                    "x_num_col_dims": self._num_flatten_dims,
                    "y_num_col_dims": 1
                })
            i += 1
            mul_results.append(tmp)

        if len(mul_results) == 1:
            pre_bias = mul_results[0]
        else:
            pre_bias = self._helper.create_variable_for_type_inference(
                self._dtype)
            self._helper.append_op(
                type="sum",
                inputs={"X": mul_results},
                outputs={"Out": pre_bias},
                attrs={"use_mkldnn": False})

        if self._b:
            pre_activation = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type="elementwise_add",
                inputs={"X": [pre_bias],
                        "Y": [self._b]},
                outputs={"Out": [pre_activation]},
                attrs={"axis": self._num_flatten_dims})
        else:
            pre_activation = pre_bias
        # Currently, we don't support inplace in dygraph mode
        return self._helper.append_activation(pre_activation, act=self._act)


class Conv2D(dg.Layer):
    """
    The convolution2D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input and
    Output are in NCHW format, where N is batch size, C is the number of
    channels, H is the height of the feature, and W is the width of the feature.
    Filter is in MCHW format, where M is the number of output image channels,
    C is the number of input image channels, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input image channels divided by the groups.
    Please refer to UFLDL's `convolution
    <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`
    for more detials.
    If bias attribution and activation type are provided, bias is added to the
    output of the convolution, and the corresponding activation function is
    applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma ((Vg) \\ast X + b)

    Where:

    * :math:`X`: Input value, a tensor with NCHW format.
    * :math:`V`: Filter direction value, a tensor with MCHW format.
    * :math:`g`: Filter magnitude value, a tensor with M format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        name_scope(str) : The name for this class.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple|None): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        stride (int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.
        padding (int|tuple): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        dilation (int|tuple): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: dilation = 1.
        groups (int): The groups number of the Conv2d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with :math:`Normal(0.0, std)`,
            and the :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None

    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
        .. code-block:: python

          from paddle.fluid.dygraph.base import to_variable
          import paddle.fluid as fluid
          from paddle.fluid.dygraph import Conv2D
          import numpy as np

          data = np.random.uniform( -1, 1, [10, 3, 32, 32] ).astype('float32')
          with fluid.dygraph.guard():
              conv2d = Conv2D( "conv2d", 2, 3)
              data = to_variable( data )
              conv = conv2d( data )

    """

    def __init__(self,
                 name_scope,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=None,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 act=None,
                 epsilon=1e-30,
                 dtype="float32"):
        assert param_attr is not False, "param_attr should not be False here."
        super(Conv2D, self).__init__(name_scope, dtype)
        self._groups = groups
        self._stride = utils.convert_to_list(stride, 2, "stride")
        self._padding = utils.convert_to_list(padding, 2, "padding")
        self._dilation = utils.convert_to_list(dilation, 2, "dilation")
        self._act = act
        if not isinstance(use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")
        self._use_cudnn = use_cudnn
        self._filter_size = filter_size
        self._num_filters = num_filters
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._epsilon = epsilon
        self._dtype = dtype
        # if (self._num_channels == self._groups and
        #         num_filters % self._num_channels == 0 and not self._use_cudnn):
        #     self._l_type = 'depthwise_conv2d'
        # else:
        # TODO(jiabin): recover the usage of depthwise_conv2d when it's
        #  kernel fixed https://github.com/PaddlePaddle/Paddle/issues/17275
        self._l_type = "conv2d"

    def _build_once(self, input):
        self._num_channels = input.shape[1]
        if self._groups is None:
            num_filter_channels = self._num_channels
        else:
            if self._num_channels % self._groups != 0:
                raise ValueError("num_channels must be divisible by groups.")
            num_filter_channels = self._num_channels // self._groups
        filter_size = utils.convert_to_list(self._filter_size, 2, "filter_size")
        filter_shape = [self._num_filters, int(num_filter_channels)
                        ] + filter_size

        def _get_default_param_initializer():
            filter_elem_num = filter_size[0] * filter_size[
                1] * self._num_channels
            std = (2.0 / filter_elem_num)**0.5
            return Normal(0.0, std, 0)

        # weight_v
        self._filter_param_v = self.create_parameter(
            attr=self._param_attr,
            shape=filter_shape,
            dtype=self._dtype,
            default_initializer=_get_default_param_initializer())

        # weight_g
        norm_value = _norm(
            self._filter_param_v.numpy(), dim=0)  # CAUTION: hard-code
        self._filter_param_g = self.create_parameter(
            attr=fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    norm_value)),
            shape=norm_value.shape,
            dtype=self._dtype,
            default_initializer=_get_default_param_initializer())

        if self._use_cudnn:
            self.create_variable(
                name="kCUDNNFwdAlgoCache",
                persistable=True,
                type=core.VarDesc.VarType.RAW)
            self.create_variable(
                name="kCUDNNBwdDataAlgoCache",
                persistable=True,
                type=core.VarDesc.VarType.RAW)
            self.create_variable(
                name="kCUDNNBwdFilterAlgoCache",
                persistable=True,
                type=core.VarDesc.VarType.RAW)

        self._bias_param = self.create_parameter(
            attr=self._bias_attr,
            shape=[self._num_filters],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, input):
        matrix = self._helper.create_variable_for_type_inference(self._dtype)
        tmp = self._helper.create_variable_for_type_inference(self._dtype)
        new_shape = [
            self._filter_param_v.shape[0],
            reduce(lambda x, y: x * y, self._filter_param_v.shape[1:], 1),
        ]

        self._helper.append_op(
            type="reshape2",
            inputs={"X": self._filter_param_v},
            attrs={"shape": new_shape},
            outputs={"Out": matrix,
                     "XShape": tmp})

        m_norm = self._helper.create_variable_for_type_inference(self._dtype)
        m_normalized = self._helper.create_variable_for_type_inference(
            self._dtype)
        self._helper.append_op(
            type="norm",
            inputs={"X": matrix},
            outputs={"Out": m_normalized,
                     "Norm": m_norm},
            attrs={"axis": 1,
                   "epsilon": self._epsilon})

        v_normalized = self._helper.create_variable_for_type_inference(
            self._dtype)
        tmp2 = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type="reshape2",
            inputs={"X": m_normalized},
            attrs={"shape": self._filter_param_v.shape},
            outputs={"Out": v_normalized,
                     "XShape": tmp2})

        filter_param = self._helper.create_variable_for_type_inference(
            self._dtype)
        self._helper.append_op(
            type="elementwise_mul",
            inputs={"X": [v_normalized],
                    "Y": [self._filter_param_g]},
            outputs={"Out": [filter_param]},
            attrs={"axis": 0},  # CAUTION: hard-code
        )

        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)

        self._helper.append_op(
            type=self._l_type,
            inputs={"Input": input,
                    "Filter": filter_param},
            outputs={"Output": pre_bias},
            attrs={
                "strides": self._stride,
                "paddings": self._padding,
                "dilations": self._dilation,
                "groups": self._groups if self._groups else 1,
                "use_cudnn": self._use_cudnn,
                "use_mkldnn": False,
            })

        if self._bias_param is not None:
            pre_act = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type="elementwise_add",
                inputs={"X": [pre_bias],
                        "Y": [self._bias_param]},
                outputs={"Out": [pre_act]},
                attrs={"axis": 1})
        else:
            pre_act = pre_bias

        # Currently, we don't support inplace in dygraph mode
        return self._helper.append_activation(pre_act, act=self._act)


class Conv2DTranspose(dg.Layer):
    """
    **Convlution2D transpose layer**

    The convolution2D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input(Input) and output(Output)
    are in NCHW format. Where N is batch size, C is the number of channels,
    H is the height of the feature, and W is the width of the feature.
    Parameters(dilations, strides, paddings) are two elements. These two elements
    represent height and width, respectively. The details of convolution transpose
    layer, please refer to the following explanation and references
    `therein <http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma ((Vg) \\ast X + b)

    Where:

    * :math:`X`: Input value, a tensor with NCHW format.
    * :math:`V`: Filter value, a tensor with MCHW format.
    * :math:`g`: Filter value, a tensor with M format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

           H^\prime_{out} &= (H_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (H_f - 1) + 1 \\\\
           W^\prime_{out} &= (W_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (W_f - 1) + 1 \\\\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[0] ) \\\\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[1] )

    Args:
        name_scope(str): The name of this class.
        num_filters(int): The number of the filter. It is as same as the output
            image channel.
        output_size(int|tuple|None): The output image size. If output size is a
            tuple, it must contain two integers, (image_H, image_W). None if use
            filter_size, padding, and stride to calculate output_size.
            if output_size and filter_size are specified at the same time, They
            should follow the formula above. Default: None.
        filter_size(int|tuple|None): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square. None if use output size to
            calculate filter_size. Default: None.
        padding(int|tuple): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        stride(int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.
        dilation(int|tuple): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: dilation = 1.
        groups(int): The groups number of the Conv2d transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups = 1.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of conv2d_transpose. If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv2d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn(bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True.
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None.

    Returns:
        Variable: The tensor variable storing the convolution transpose result.

    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
       .. code-block:: python

          import paddle.fluid as fluid
          import numpy

          with fluid.dygraph.guard():
              data = numpy.random.random((3, 32, 32)).astype('float32')
              conv2DTranspose = fluid.dygraph.nn.Conv2DTranspose(
                    'Conv2DTranspose', num_filters=2, filter_size=3)
              ret = conv2DTranspose(fluid.dygraph.base.to_variable(data))

    """

    def __init__(self,
                 name_scope,
                 num_filters,
                 output_size=None,
                 filter_size=None,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=None,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 epsilon=1e-30,
                 act=None,
                 dtype="float32"):
        super(Conv2DTranspose, self).__init__(name_scope, dtype)
        assert (param_attr is not False
                ), "param_attr should not be False in conv2d_transpose."
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._groups = groups
        self._num_filters = num_filters
        self._use_cudnn = use_cudnn
        self._padding = padding
        self._stride = stride
        self._dilation = dilation
        self._filter_size = filter_size
        self._output_size = output_size
        self._op_type = "conv2d_transpose"
        self._epsilon = epsilon

    def _build_once(self, input):
        input_channel = input.shape[1]
        if (input_channel == self._groups and
                self._num_filters == input_channel and not self._use_cudnn):
            self._op_type = "depthwise_conv2d_transpose"

        if not isinstance(input, Variable):
            raise TypeError("Input of conv2d_transpose must be Variable")

        self._padding = utils.convert_to_list(self._padding, 2, "padding")
        self._stride = utils.convert_to_list(self._stride, 2, "stride")
        self._dilation = utils.convert_to_list(self._dilation, 2, "dilation")

        if not isinstance(self._use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")

        if self._filter_size is None:
            if self._output_size is None:
                raise ValueError(
                    "output_size must be set when filter_size is None")
            if isinstance(self._output_size, int):
                self._output_size = [self._output_size, self._output_size]

            h_in = input.shape[2]
            w_in = input.shape[3]

            filter_size_h = (self._output_size[0] -
                             (h_in - 1) * self._stride[0] + 2 * self._padding[0]
                             - 1) // self._dilation[0] + 1
            filter_size_w = (self._output_size[1] -
                             (w_in - 1) * self._stride[1] + 2 * self._padding[1]
                             - 1) // self._dilation[1] + 1
            self._filter_size = [filter_size_h, filter_size_w]
        else:
            self._filter_size = utils.convert_to_list(
                self._filter_size, 2, "conv2d_transpose.filter_size")

        if self._output_size is None:
            self._output_size = []
        elif isinstance(self._output_size, list) or isinstance(
                self._output_size, int):
            self._output_size = utils.convert_to_list(self._output_size, 2,
                                                      "output_size")
        else:
            raise ValueError("output_size should be list or int")
        self._padding = utils.convert_to_list(self._padding, 2, "padding")
        self._groups = 1 if self._groups is None else self._groups
        filter_shape = [
            input_channel,
            self._num_filters // self._groups,
        ] + self._filter_size

        # img filter v (direction)
        self._img_filter_v = self.create_parameter(
            dtype=input.dtype, shape=filter_shape, attr=self._param_attr)

        # img filter g (magnitude)
        img_filter_magnitude = _norm(
            self._img_filter_v.numpy(), dim=0)  # CAUTION: hard-code
        self._img_filter_g = self.create_parameter(
            dtype=input.dtype,
            shape=img_filter_magnitude.shape,
            attr=fluid.ParamAttr(
                initializer=NumpyArrayInitializer(img_filter_magnitude)))

        self._img_bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[self._num_filters],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, input):
        matrix = self._helper.create_variable_for_type_inference(self._dtype)
        tmp = self._helper.create_variable_for_type_inference(self._dtype)
        new_shape = [
            self._img_filter_v.shape[0],
            reduce(lambda x, y: x * y, self._img_filter_v.shape[1:], 1),
        ]

        self._helper.append_op(
            type="reshape2",
            inputs={"X": self._img_filter_v},
            attrs={"shape": new_shape},
            outputs={"Out": matrix,
                     "XShape": tmp})

        m_norm = self._helper.create_variable_for_type_inference(self._dtype)
        m_normalized = self._helper.create_variable_for_type_inference(
            self._dtype)
        self._helper.append_op(
            type="norm",
            inputs={"X": matrix},
            outputs={"Out": m_normalized,
                     "Norm": m_norm},
            attrs={"axis": 1,
                   "epsilon": self._epsilon})

        v_normalized = self._helper.create_variable_for_type_inference(
            self._dtype)
        tmp2 = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type="reshape2",
            inputs={"X": m_normalized},
            attrs={"shape": self._img_filter_v.shape},
            outputs={"Out": v_normalized,
                     "XShape": tmp2})

        img_filter = self._helper.create_variable_for_type_inference(
            self._dtype)
        self._helper.append_op(
            type="elementwise_mul",
            inputs={"X": [v_normalized],
                    "Y": [self._img_filter_g]},
            outputs={"Out": [img_filter]},
            attrs={"axis": 0},  # CAUTION: hard-code
        )

        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=input.dtype)
        self._helper.append_op(
            type=self._op_type,
            inputs={"Input": [input],
                    "Filter": [img_filter]},
            outputs={"Output": pre_bias},
            attrs={
                "output_size": self._output_size,
                "strides": self._stride,
                "paddings": self._padding,
                "dilations": self._dilation,
                "groups": self._groups,
                "use_cudnn": self._use_cudnn,
            })

        if self._img_bias is not None:
            pre_act = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type="elementwise_add",
                inputs={"X": [pre_bias],
                        "Y": [self._img_bias]},
                outputs={"Out": [pre_act]},
                attrs={"axis": 1})
        else:
            pre_act = pre_bias

        out = self._helper.append_activation(pre_act)
        return out
