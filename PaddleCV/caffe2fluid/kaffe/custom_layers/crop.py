""" a custom layer for 'crop', maybe we should implement this in standard way.
    more info can be found here: http://caffe.berkeleyvision.org/tutorial/layers/crop.html
"""
from .register import register


def crop_shape(input_shape, shape=None):
    """ calculate the output shape of this layer using input shape

    Args:
        @input_shape (num | list of num): a list of number or num which represents the input shape
        @shape (list of integer): the shape of output

    Returns:
        @output_shape (list of num): a list of numbers represent the output shape
    """
    if isinstance(input_shape, list):
        assert len(input_shape) == 2, "the number of crop's inputs must be 2"
        return input_shape[1]
    elif not shape is None:
        assert len(shape) == len(
            input_shape.shape), "input_shape is diff with output_shape"
        return shape
    else:
        raise Exception, "crop_shape input error"
        return None


def crop_layer(input, name, shape=None, axis=2, offset=None):
    """ build a layer of type 'Crop' using fluid

    Args:
        @input (variables | list of variables): input fluid variable for this layer
        @shape (list of integer): the shape of output
        @name (str): name for this layer
        @axis (integer): parameter from caffe's Crop layer
        @offset (Variable|list/tuple of integer|None): parameter from caffe's Crop layer

    Returns:
        output (variable): output variable for this layer
    """
    input_shape = None
    output_shape = None
    input_tensor = None
    if isinstance(input, list):
        assert len(input) == 2, "the number of crop's inputs must be 2"
        input_shape = input[0].shape
        output_shape = input[1].shape
        input_tensor = input[0]
    elif not shape is None:
        assert len(shape) == len(
            input.shape), "input_shape is diff with output_shape"
        input_shape = input.shape
        output_shape = shape
        input_tensor = input
    else:
        raise Exception, "crop_layer input error"

    assert len(output_shape) == len(
        input_shape), "input_shape is diff with output_shape"

    if axis < 0:
        axis += len(input_shape)

    if offset is not None:
        assert (len(input_shape) - axis
                ) == len(offset), "invalid offset[%s] in crop layer" % (
                    str(offset))
        offset = [0] * axis + offset
    import paddle.fluid as fluid
    output = fluid.layers.crop(
        input_tensor, shape=output_shape, offsets=offset, name=name)

    return output


register(kind='Crop', shape=crop_shape, layer=crop_layer)
