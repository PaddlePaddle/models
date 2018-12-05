""" a custom layer for 'crop', maybe we should implement this in standard way.
    more info can be found here: http://caffe.berkeleyvision.org/tutorial/layers/reduction.html
"""
from .register import register


def reduction_shape(input_shape, axis=0):
    """ calculate the output shape of this layer using input shape

    Args:
        @input_shape (list of num): a list of number which represents the input shape
        @axis (int): parameter from caffe's reduction layer

    Returns:
        @output_shape (list of num): a list of numbers represent the output shape
    """
    if axis < 0:
        axis += len(input_shape) + 1

    assert axis <= len(input_shape), 'invalid axis[%d] error' % (axis)

    return input_shape[0:axis]


def reduction_layer(input, name, axis=0, operation=1, coeff=1.0):
    """ build a layer of type 'Crop' using fluid

    Args:
        @input (variable): input fluid variable for this layer
        @name (str): name for this layer
        @axis (int): parameter from caffe's reduction layer
        @operation (int): parameter from caffe's reduction layer
        @coeff (float): parameter from caffe's reduction layer

    Returns:
        output (variable): output variable for this layer
    """
    assert operation >= 1 and operation <= 4, "reduction reduction [%s] error" % (
        operation)

    input_len = len(input.shape)
    if axis < 0:
        axis += input_len + 1

    dim = range(input_len)

    import paddle.fluid as fluid
    if operation == 1:  ## operation = SUM
        output = fluid.layers.reduce_sum(
            input, dim=dim[axis:], keep_dim=False, name=name)
    elif operation == 2:  ## operation = ASUM
        absout = fluid.layers.abs(input)
        output = fluid.layers.reduce_sum(
            absout, dim=dim[axis:], keep_dim=False, name=name)
    elif operation == 3:  ## operation = SUMSQ
        powout = fluid.layers.pow(x=input, factor=2.0)
        output = fluid.layers.reduce_sum(
            powout, dim=dim[axis:], keep_dim=False, name=name)
    else:  ## operation = MEAN
        output = fluid.layers.reduce_mean(
            input, dim=dim[axis:], keep_dim=False, name=name)

    mulout = fluid.layers.scale(x=output, scale=coeff)
    return mulout


register(kind='Reduction', shape=reduction_shape, layer=reduction_layer)
