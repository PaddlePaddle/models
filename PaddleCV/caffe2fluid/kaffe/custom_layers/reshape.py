""" a custom layer for 'reshape', maybe we should implement this in standard way.
    more info can be found here: http://caffe.berkeleyvision.org/tutorial/layers/reshape.html
"""
from .register import register


def import_fluid():
    import paddle.fluid as fluid
    return fluid


def reshape_shape(input_sp, shape, axis=0, num_axes=-1):
    """ calculate the output shape of this layer using input shape

    Args:
        @input_shape (list of num): a list of number which represents the input shape
        @shape (object): parameter from caffe's Reshape layer
        @axis (int): parameter from caffe's Reshape layer
        @num_axes(int): parameter from caffe's Reshape layer

    Returns:
        @output_shape (list of num): a list of numbers represent the output shape
    """

    def count(num_list):
        return reduce(lambda a, b: a * b, num_list)

    input_shape = list(input_sp)
    input_count = count(input_shape)

    input_num_axes = len(input_shape)

    input_start_axis = axis
    start_axis = input_start_axis if input_start_axis >= 0 \
            else input_num_axes + input_start_axis + 1

    assert start_axis >= 0, "[Reshape]axis %d out of range" % (input_start_axis)
    assert start_axis <= input_num_axes, "[Reshape]axis %d out of range for %d-D input data"\
            % (input_start_axis, input_num_axes)

    assert num_axes >= -1, "[Reshape]num_axes must be >= 0, or -1 for all"

    end_axis = input_num_axes if num_axes == -1 else start_axis + num_axes
    assert end_axis <= input_num_axes, "end_axis[%d] = axis[%d] + num_axes[%d] is out of range"\
            % (end_axis, start_axis, num_axes)

    num_axes_replaced = end_axis - start_axis
    num_axes_retained = input_num_axes - num_axes_replaced
    num_new_axes = len(shape['dim'])
    output_shape = []

    for i in range(start_axis):
        output_shape.append(input_shape[i])

    for i in range(num_new_axes):
        output_shape.append(shape['dim'][i])

    for i in range(end_axis, input_num_axes):
        output_shape.append(input_shape[i])

    assert len(output_shape) == num_axes_retained + num_new_axes,\
            "[Reshape]invalid dims of output shape[%s]" % (str(output_shape))

    inferred_axis = -1
    copy_axes = []
    constant_count = 1
    for i in range(num_new_axes):
        top_dim = shape['dim'][i]
        if top_dim == 0:
            copy_axes.append(i)
            copy_axis_index = start_axis + i
            output_shape[copy_axis_index] = input_shape[copy_axis_index]
        elif top_dim == -1:
            assert inferred_axis == -1, "[Reshape]new shape contains multiple -1 dims"
            inferred_axis = i
        else:
            constant_count *= top_dim

    if inferred_axis >= 0:
        explicit_count = constant_count
        l = input_shape[0:start_axis]
        if len(l) > 0:
            explicit_count *= count(l)

        l = input_shape[end_axis:]
        if len(l) > 0:
            explicit_count *= count(l)

        for i in range(len(copy_axes)):
            explicit_count *= output_shape[start_axis + copy_axes[i]]

        assert input_count % explicit_count == 0, "[Reshape]botom count[%d] "\
                "must be divisible by product of the specified dimensions[%d] "\
                % (input_count, explicit_count)
        output_shape[start_axis + inferred_axis] = input_count / explicit_count

    output_count = count(output_shape)
    assert output_count == input_count, "[Reshape]output count[%d] must match input count[%d]" % (
        output_count, input_count)

    return output_shape


def reshape_layer(input, name, shape, axis=0, num_axes=-1):
    """ build a layer of type 'Flatten' using fluid

    Args:
        @input (variable): input fluid variable for this layer
        @name (str): name for this layer
        @shape (object): parameter from caffe's Reshape layer
        @axis (int): parameter from caffe's Reshape layer
        @num_axes(int): parameter from caffe's Reshape layer

    Returns:
        output (variable): output variable for this layer
    """
    fluid = import_fluid()

    input_shape = list(input.shape)

    if input_shape[0] == -1:
        input_shape[0] = 1
        output_shape = reshape_shape(input_shape, shape, axis, num_axes)
        output_shape[0] = -1
    else:
        output_shape = reshape_shape(input_shape, shape, axis, num_axes)

    output = fluid.layers.reshape(input, shape=output_shape, name=name)

    return output


register(kind='Reshape', shape=reshape_shape, layer=reshape_layer)
