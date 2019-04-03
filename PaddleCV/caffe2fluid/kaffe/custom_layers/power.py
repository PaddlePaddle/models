""" a custom layer for 'power', maybe we should implement this in standard way.
    more info can be found here: http://caffe.berkeleyvision.org/tutorial/layers/power.html
"""
from .register import register


def power_shape(input_shape, shape=None):
    """ calculate the output shape of this layer using input shape

    Args:
        @input_shape (list of num): a list of number which represents the input shape

    Returns:
        @output_shape (list of num): a list of numbers represent the output shape
    """
    return input_shape


def power_layer(input, name, power=1.0, scale=1.0, shift=0.0):
    """ build a layer of type 'Power' using fluid

    Args:
        @input (variables): input fluid variable for this layer
        @name (str): name for this layer
        @power (float): parameter from caffe's Power layer
	@scale (float): parameter from caffe's Power layer
        @shift (float): parameter from caffe's Power layer

    Returns:
        output (variable): output variable for this layer
    """
    import paddle.fluid as fluid
    scale_out = fluid.layers.scale(
        input, scale=scale, bias=shift, bias_after_scale=True)
    output = fluid.layers.pow(scale_out, factor=power)

    return output


register(kind='Power', shape=power_shape, layer=power_layer)
