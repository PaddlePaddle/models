""" A custom layer for 'Permute' which is equivalent to transpose in paddle
"""

from .register import register


def permute_shape(input_shape, order):
    """ calculate the output shape of this layer using input shapes

    Args:
        @input_shape (list of numbers): input shape

    Returns:
        @output_shape (list of num): a list of numbers represent the output shape
    """
    output_shape = []
    for ii in order:
        assert ii < len(input_shape), "invalid order for permute[%s]" % (name)
        output_shape.append(input_shape[ii])
    return output_shape


def permute_layer(input, name, order):
    """ build a layer of type 'permute' using fluid

    Args:
        @input (input variable): input fluid variables for this layer
        @name (str): name for this layer
        @order (list of int): order to permute the dims

    Returns:
        output (variable): output variable for this layer
    """
    import paddle.fluid as fluid
    output = fluid.layers.transpose(input, order, name=name)

    return output


register(kind='Permute', shape=permute_shape, layer=permute_layer)
