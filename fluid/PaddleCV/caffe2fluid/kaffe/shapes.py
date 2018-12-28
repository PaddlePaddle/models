import math
from collections import namedtuple

from .errors import KaffeError

Tensor4DShape = namedtuple('Tensor4DShape',
                           ['batch_size', 'channels', 'height', 'width'])

Tensor3DShape = namedtuple('Tensor3DShape', ['batch_size', 'data1', 'data2'])

Tensor2DShape = namedtuple('Tensor2DShape', ['batch_size', 'data'])

ScalarShape = namedtuple('ScalarShape', ['batch_size'])


def make_tensor(batch_size, d1=None, d2=None, d3=None):
    if d3 is not None:
        return Tensor4DShape(batch_size, d1, d2, d3)
    elif d1 is not None and d2 is not None:
        return Tensor3DShape(batch_size, d1, d2)
    elif d1 is not None and d2 is None:
        return Tensor2DShape(batch_size, d1)
    elif d1 is None and d2 is None and d3 is None:
        return ScalarShape(batch_size)
    else:
        raise NotImplementedError('invalid params for make_tensor %s' \
                % (str((batch_size, d1, d2, d3))))


def get_filter_output_shape(i_h, i_w, params, round_func):
    dila_h = getattr(params, 'dila_h', 1)
    dila_w = getattr(params, 'dila_w', 1)

    o_h = (i_h + 2 * params.pad_h -
           (dila_h * (params.kernel_h - 1) + 1)) / float(params.stride_h) + 1
    o_w = (i_w + 2 * params.pad_w -
           (dila_w * (params.kernel_w - 1) + 1)) / float(params.stride_w) + 1

    return (int(round_func(o_h)), int(round_func(o_w)))


def get_strided_kernel_output_shape(node, round_func):
    assert node.layer is not None
    input_shape = node.get_only_parent().output_shape
    o_h, o_w = get_filter_output_shape(input_shape.height, input_shape.width,
                                       node.layer.kernel_parameters, round_func)
    params = node.layer.parameters
    has_c_o = hasattr(params, 'num_output')
    c = params.num_output if has_c_o else input_shape.channels
    return make_tensor(input_shape.batch_size, c, o_h, o_w)


def shape_not_implemented(node):
    raise NotImplementedError


def shape_identity(node):
    assert len(node.parents) > 0
    return node.parents[0].output_shape


def shape_scalar(node):
    return make_tensor(1, 1, 1, 1)


def shape_crop(node):
    raise KaffeError('crop function had been defined in customer_layers')


def shape_power(node):
    raise KaffeError('power function had been defined in customer_layers')


def shape_data(node):
    if node.output_shape:
        # Old-style input specification
        shape = node.output_shape
    else:
        try:
            # New-style input specification
            shape = map(int, node.parameters.shape[0].dim)
        except:
            # We most likely have a data layer on our hands. The problem is,
            # Caffe infers the dimensions of the data from the source (eg: LMDB).
            # We want to avoid reading datasets here. Fail for now.
            # This can be temporarily fixed by transforming the data layer to
            # Caffe's "input" layer (as is usually used in the "deploy" version).
            # TODO: Find a better solution for this.
            raise KaffeError(
                'Cannot determine dimensions of data layer.\n'
                'See comments in function shape_data for more info.')
    return shape


def shape_mem_data(node):
    params = node.parameters
    return make_tensor(params.batch_size, params.channels, params.height,
                       params.width)


def shape_concat(node):
    axis = node.layer.parameters.axis
    output_shape = None
    for parent in node.parents:
        if output_shape is None:
            output_shape = list(parent.output_shape)
        else:
            output_shape[axis] += parent.output_shape[axis]
    return tuple(output_shape)


def shape_convolution(node):
    return get_strided_kernel_output_shape(node, math.floor)


def shape_deconvolution(node):
    assert node.layer is not None
    input_shape = node.get_only_parent().output_shape
    h_i = input_shape.height
    w_i = input_shape.width

    params = node.layer.kernel_parameters
    p_h = params.pad_h
    p_w = params.pad_w

    dila_h = params.dila_h
    dila_w = params.dila_w

    k_h = params.kernel_h
    k_w = params.kernel_w

    s_h = params.stride_h
    s_w = params.stride_w

    h_o = (h_i - 1) * s_h - 2 * p_h + dila_h * (k_h - 1) + 1
    w_o = (w_i - 1) * s_w - 2 * p_w + dila_w * (k_w - 1) + 1

    params = node.layer.parameters
    has_c_o = hasattr(params, 'num_output')
    c = params.num_output if has_c_o else input_shape.channels
    return make_tensor(input_shape.batch_size, c, h_o, w_o)


def shape_pool(node):
    global_pool = getattr(node.layer.parameters, 'global_pooling', False)
    if global_pool:
        input_shape = node.get_only_parent().output_shape
        return make_tensor(input_shape.batch_size, input_shape.channels, 1, 1)

    ceil_mode = getattr(node.layer.parameters, 'ceil_mode', True)
    if ceil_mode is True:
        method = math.ceil
    else:
        method = math.floor
    return get_strided_kernel_output_shape(node, method)


def shape_inner_product(node):
    input_shape = node.get_only_parent().output_shape
    return make_tensor(input_shape.batch_size, node.layer.parameters.num_output)
