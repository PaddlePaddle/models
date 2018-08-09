""" a custom layer for 'select' which is used to replace standard 'Slice' layer 
    for converting layer with multiple different output tensors
"""
from .register import register


def select_shape(input_shape, slice_point, axis=1):
    """ calculate the output shape of this layer using input shape

    Args:
        @input_shape (list of num): a list of number which represents the input shape
        @slice_point (list): parameter from caffe's Slice layer
        @axis (int): parameter from caffe's Slice layer

    Returns:
        @output_shape (list of num): a list of numbers represent the output shape
    """

    input_shape = list(input_shape)
    start = slice_point[0]
    if len(slice_point) == 2:
        end = slice_point[1]
    else:
        end = input_shape[axis]

    assert end > start, "invalid slice_point with [start:%d, end:%d]"\
             % (start, end)
    output_shape = input_shape
    output_shape[axis] = end - start
    return output_shape


def select_layer(input, name, slice_point, axis=1):
    """ build a layer of type 'Slice' using fluid

    Args:
        @input (variable): input fluid variable for this layer
        @name (str): name for this layer
        @slice_point (list): parameter from caffe's Slice layer
        @axis (int): parameter from caffe's Slice layer

    Returns:
        output (variable): output variable for this layer
    """
    import paddle.fluid as fluid
    input_shape = list(input.shape)

    start = slice_point[0]
    if len(slice_point) == 2:
        end = slice_point[1]
    else:
        end = input_shape[axis]

    sections = []
    if start > 0:
        sections.append(start)

    pos = len(sections)
    sections.append(end - start)
    if end != input_shape[axis]:
        sections.append(input_shape[axis] - end)

    outputs = fluid.layers.split(input, sections, dim=axis, name=name)
    return outputs[pos]


register(kind='Select', shape=select_shape, layer=select_layer)
