from __future__ import print_function
import paddle
import paddle.fluid as fluid

from paddle.fluid import core

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.layer_object_helper import LayerObjectHelper

from paddle.fluid.initializer import Normal, Constant, NumpyArrayInitializer

from paddle.fluid.param_attr import ParamAttr

from paddle.fluid.framework import Variable, OpProtoHolder, in_dygraph_mode
from paddle.fluid.layers import utils
import numpy as np


def Fconv2d(input,
            filter,
            stride=1,
            padding=0,
            dilation=1,
            groups=None,
            use_cudnn=True,
            name=None):
    """
    Similar with conv2d, this is a convolution2D layers. Difference
    is filter can be token as input directly instead of setting filter size
    and number of fliters. Filter is a  4-D tensor with shape
    [num_filter, num_channel, filter_size_h, filter_size_w].
     Args:
        input (Variable): The input image with [N, C, H, W] format.
        filter(Variable): The input filter with [out_channels, in_channels, H, W] format.
        stride (int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.
        padding (int|tuple): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        dilation (int|tuple): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: dilation = 1.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: None
    Returns:
        Variable: The tensor variable storing the convolution and \
                  non-linearity activation result.
    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.
    Examples:
        .. code-block:: python
          data = fluid.layers.data(name='data', shape=[3, 32, 32], \
                                  dtype='float32')
          filter = fluid.layers.data(name='filter',shape=[10,3,3,3], \
                                    dtype='float32',append_batch_size=False)
          conv2d = fluid.layers.conv2d(input=data,
                                       filter=filter,
                                       act="relu")
    """
    helper = LayerHelper("conv2d_with_filter", **locals())
    num_channels = input.shape[1]
    num_filters = filter.shape[0]
    num_filter_channels = filter.shape[1]
    l_type = 'conv2d'
    # if (num_channels == groups and
    if (num_channels == groups and num_filters % num_channels == 0 and
            not use_cudnn):
        l_type = 'depthwise_conv2d'
    if groups is None:
        assert num_filter_channels == num_channels
        groups = 1
    else:
        if num_channels % groups != 0:
            raise ValueError("num_channels must be divisible by groups.")
        if num_channels // groups != num_filter_channels:
            raise ValueError("num_filter_channels must equal to num_channels\
                              divided by groups.")

    stride = utils.convert_to_list(stride, 2, 'stride')
    padding = utils.convert_to_list(padding, 2, 'padding')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')
    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")
    pre_bias = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=l_type,
        inputs={
            'Input': input,
            'Filter': filter,
        },
        outputs={"Output": pre_bias},
        attrs={
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': False
        })

    return pre_bias


def test_conv2d_with_filter():
    exemplar = np.random.random((8, 4, 6, 6)).astype(np.float32)
    instance = np.random.random((8, 4, 22, 22)).astype(np.float32)

    # fluid.layers.data(append_batch_size=)
    use_gpu = False
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

    train_program = fluid.Program()
    start_program = fluid.Program()

    with fluid.program_guard(train_program, start_program):
        x = fluid.layers.data(
            name="inst", shape=[8, 4, 22, 22], append_batch_size=False)
        y = fluid.layers.data(
            name="exem", shape=[8, 4, 6, 6], append_batch_size=False)
        bias_att = fluid.ParamAttr(
            name="bias_", initializer=fluid.initializer.ConstantInitializer(1.))
        out = conv2d_with_filter(x, y, groups=1)
        weight_att = fluid.ParamAttr(
            name='weight',
            initializer=fluid.initializer.NumpyArrayInitializer(exemplar))
        bias_att = fluid.ParamAttr(
            name="bias", initializer=fluid.initializer.ConstantInitializer(0.))
        res = fluid.layers.conv2d(
            x,
            8,
            6,
            param_attr=weight_att,
            bias_attr=bias_att,
            stride=1,
            padding=0,
            dilation=1)

        exe = fluid.Executor(place)
        exe.run(program=fluid.default_startup_program())
    print(out.shape)

    compiled_prog = fluid.compiler.CompiledProgram(train_program)
    out, res = exe.run(compiled_prog,
                       feed={"inst": instance,
                             "exem": exemplar},
                       fetch_list=[out.name, res.name])

    print(np.sum(out - res))
    np.testing.assert_allclose(out, res, rtol=1e-5, atol=0)

    with fluid.dygraph.guard():
        exem = fluid.dygraph.to_variable(exemplar)
        inst = fluid.dygraph.to_variable(instance)

        out = conv2d_with_filter(inst, exem, groups=1)

    print(np.sum(out.numpy() - res))
    np.testing.assert_allclose(out.numpy(), res, rtol=1e-5, atol=0)


if __name__ == '__main__':
    test_conv2d_with_filter()
