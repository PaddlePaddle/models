""" a custom layer for 'argmax', maybe we should implement this in standard way.
    more info can be found here: http://caffe.berkeleyvision.org/tutorial/layers/argmax.html
"""
from .register import register


def import_fluid():
    import paddle.fluid as fluid
    return fluid


def argmax_shape(input_shape, out_max_val=False, top_k=1, axis=-1):
    """ calculate the output shape of this layer using input shape

    Args:
        @input_shape (list of num): a list of number which represents the input shape
        @out_max_val (bool): parameter from caffe's ArgMax layer
        @top_k (int): parameter from caffe's ArgMax layer
        @axis (int): parameter from caffe's ArgMax layer

    Returns:
        @output_shape (list of num): a list of numbers represent the output shape
    """
    input_shape = list(input_shape)

    if axis < 0:
        axis += len(input_shape)

    assert (axis + 1 == len(input_shape)
            ), 'only can be applied on the last dimension now'
    """
    #more general implementation
    for i in range(len(input_shape)):
        if i == axis:
            continue
        else:
            output_shape.append(input_shape[i])
    """

    output_shape = input_shape
    output_shape[-1] = top_k

    return output_shape


def argmax_layer(input, name, out_max_val=False, top_k=1, axis=-1):
    """ build a layer of type 'ArgMax' using fluid

    Args:
        @input (variable): input fluid variable for this layer
        @name (str): name for this layer
        @out_max_val (bool): parameter from caffe's ArgMax layer
        @top_k (int): parameter from caffe's ArgMax layer
        @axis (int): parameter from caffe's ArgMax layer

    Returns:
        output (variable): output variable for this layer
    """

    fluid = import_fluid()

    if axis < 0:
        axis += len(input.shape)

    assert (axis + 1 == len(input_shape)
            ), 'only can be applied on the last dimension now'

    output = fluid.layers.topk(input=input, k=top_k)
    return output


register(kind='ArgMax', shape=argmax_shape, layer=argmax_layer)
