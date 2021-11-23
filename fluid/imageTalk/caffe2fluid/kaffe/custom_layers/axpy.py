""" A custom layer for 'axpy' which receives 3 tensors and output 1 tensor.
    the function performed is:(the mupltiplication and add are elementewise)
        output = inputs[0] * inputs[1] + inputs[2]
"""

from .register import register


def axpy_shape(input_shapes):
    """ calculate the output shape of this layer using input shapes

    Args:
        @input_shapes (list of tuples): a list of input shapes

    Returns:
        @output_shape (list of num): a list of numbers represent the output shape
    """
    assert len(input_shapes) == 3, "not valid input shape for axpy layer"
    assert len(input_shapes[0]) == len(input_shapes[1]), 'should have same dims'

    output_shape = input_shapes[1]
    assert (input_shapes[2] == output_shape),\
            "shape not consistent for axpy[%s <--> %s]" \
            % (str(output_shape), str(input_shapes[2]))

    return output_shape


def axpy_layer(inputs, name):
    """ build a layer of type 'Axpy' using fluid

    Args:
        @inputs (list of variables): input fluid variables for this layer
        @name (str): name for this layer

    Returns:
        output (variable): output variable for this layer
    """
    import paddle.fluid as fluid

    assert len(inputs) == 3, "invalid inputs for axpy[%s]" % (name)
    alpha = inputs[0]
    x = inputs[1]
    y = inputs[2]
    output = fluid.layers.elementwise_mul(x, alpha, axis=0)
    output = fluid.layers.elementwise_add(output, y, name=name)

    return output


register(kind='Axpy', shape=axpy_shape, layer=axpy_layer)
