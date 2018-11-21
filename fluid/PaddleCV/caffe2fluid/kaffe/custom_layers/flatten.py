""" a custom layer for 'flatten', maybe we should implement this in standard way.
    more info can be found here: http://caffe.berkeleyvision.org/tutorial/layers/flatten.html
"""
from .register import register


def flatten_shape(input_shape, axis=1, end_axis=-1):
    """ calculate the output shape of this layer using input shape

    Args:
        @input_shape (list of num): a list of number which represents the input shape
        @axis (int): parameter from caffe's Flatten layer
        @end_axis (int): parameter from caffe's Flatten layer

    Returns:
        @output_shape (list of num): a list of numbers represent the output shape
    """

    start_axis = axis
    end_axis = end_axis
    input_shape = list(input_shape)
    if start_axis < 0:
        start_axis += len(input_shape)

    if end_axis < 0:
        end_axis += len(input_shape) + 1

    assert start_axis <= end_axis, 'invalid axis[%d] or end_axis[%d] params'\
            % (start_axis, end_axis)
    output_shape = input_shape[0:start_axis]
    flat_sz = reduce(lambda a, b: a * b, input_shape[start_axis:end_axis])
    output_shape += [flat_sz]
    output_shape += input_shape[end_axis:-1]

    return output_shape


def flatten_layer(input, name, axis=1, end_axis=-1):
    """ build a layer of type 'Flatten' using fluid

    Args:
        @input (variable): input fluid variable for this layer
        @name (str): name for this layer
        @axis (int): parameter from caffe's Flatten layer
        @end_axis (int): parameter from caffe's Flatten layer

    Returns:
        output (variable): output variable for this layer
    """
    import paddle.fluid as fluid

    input_shape = list(input.shape)

    if input_shape[0] == -1:
        input_shape[0] = 1
        output_shape = flatten_shape(input_shape, axis=axis, end_axis=end_axis)
        output_shape[0] = -1
    else:
        output_shape = flatten_shape(input_shape, axis=axis, end_axis=end_axis)

    output = fluid.layers.reshape(input, shape=output_shape, name=name)

    return output


register(kind='Flatten', shape=flatten_shape, layer=flatten_layer)
