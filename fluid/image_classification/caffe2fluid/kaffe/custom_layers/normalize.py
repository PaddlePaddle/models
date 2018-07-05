""" A custom layer for 'normalize' op
"""

from .register import register


def normalize_shape(input_shape,
                    across_spatial=True,
                    scale_filler=True,
                    eps=1e-10):
    """ calculate the output shape of this layer using input shapes

    Args:
        @input_shape (list of tuples): input shape

    Returns:
        @output_shape (list of num): a list of numbers represent the output shape
    """
    output_shape = input_shape
    return output_shape


def normalize_layer(input,
                    name,
                    across_spatial=True,
                    scale_filler=True,
                    channel_shared=False,
                    eps=1e-10):
    """ build a layer of type 'normalize' using fluid

    Args:
        @inputs (list of variables): input fluid variables for this layer
        @name (str): name for this layer

    Returns:
        output (variable): output variable for this layer
    """
    import paddle.fluid as fluid

    param_prefix = name.split('.')[0]

    assert across_spatial == False, "Only support across_spatial == False for Normalize[%s]" % (
        name)
    l2_norm = fluid.layers.l2_normalize(input, axis=1)  # l2 norm along channel

    input_shape = list(input.shape)
    if channel_shared:
        shape = [1]
    else:
        shape = [1] * len(input_shape)
        shape[-1] = input.shape[1]

    #shape = [1] if channel_shared else [input.shape[1]]
    scale_attr = fluid.ParamAttr(name=param_prefix + '_scale')
    scale_param = fluid.layers.create_parameter(
        shape=shape, dtype=input.dtype, name=name, attr=scale_attr)

    scale_y = fluid.layers.reshape(scale_param, shape=[512])
    out = fluid.layers.elementwise_mul(
        x=l2_norm, y=scale_y, axis=-1 if channel_shared else 1)
    return out


register(kind='Normalize', shape=normalize_shape, layer=normalize_layer)
