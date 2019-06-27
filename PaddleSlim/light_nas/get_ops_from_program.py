from __future__ import print_function
import numpy as np


def conv_op_params(blocks, current_op):
    """Getting params of conv op
    Args:
        blocks, current block
        current_op, current op
    Returns:
        list, op name and hyperparamters
    """
    tmp, res = [], []
    # op_name
    tmp.append('conv')
    # cluster, threads, test_iter
    tmp = tmp + [0, 1, 100]
    # flag_bias
    if not current_op.input('Bias'):
        tmp.append(0)
    else:
        tmp.append(1)
    # flag_relu
    tmp.append(int(current_op.attr('fuse_relu')))
    # batch size
    tmp.append(1)
    # channels, height, width
    in_shapes = blocks.vars[current_op.input('Input')[0]].shape
    tmp = tmp + [int(in_shapes[1]), int(in_shapes[2]), int(in_shapes[3])]

    # output channels
    w_shapes = blocks.vars[current_op.input('Filter')[0]].shape
    tmp.append(int(w_shapes[0]))

    # group
    tmp.append(int(current_op.attr('groups')))

    # kernel size
    tmp.append(int(w_shapes[2]))
    if w_shapes[2] != w_shapes[3]:
        res.append(int(w_shapes[3]))

    # padding
    paddings = current_op.attr('paddings')
    tmp.append(int(paddings[0]))
    if paddings[0] != paddings[1]:
        res.append(int(paddings[0]))

    # strides
    strides = current_op.attr('strides')
    tmp.append(int(strides[0]))
    if strides[0] != strides[1]:
        res.append(int(strides[1]))

    # dilations
    dilations = current_op.attr('dilations')
    tmp.append(int(dilations[0]))
    if dilations[0] != dilations[1]:
        res.append(int(dilations[1]))

    tmp = tmp + res
    return tmp


def batch_norm_op_params(blocks, current_op):
    """Getting params of batch_norm op
    Args:
        blocks, current block
        current_op, current op
    Returns:
        list, op name and hyperparamters
    """
    tmp = []
    # op name
    tmp.append('batch_norm')
    # cluster, threads, test_iters
    tmp = tmp + [0, 1, 100]
    # activation type
    if not current_op.attr('fuse_with_relu'):
        tmp.append('None')
    else:
        tmp.append('relu')
    # batch size
    tmp.append(1)
    # input channels, height, width
    in_shapes = blocks.vars[current_op.input("X")[0]].shape
    tmp = tmp + [int(in_shapes[1]), int(in_shapes[2]), int(in_shapes[3])]
    return tmp


def eltwise_op_params(blocks, current_op):
    """Getting params of eltwise op
    Args:
        blocks, current block
        current_op, current op
    Returns:
        list, op name and hyperparamters
    """
    # op name, clusters, threads, test_iter
    tmp = ['eltwise', 0, 1, 100]
    # elementwise type, TODO: add more ops
    if current_op.type == 'elementwise_add':
        tmp.append(1)
    elif current_op.type == 'elementwise_mul':
        tmp.append(2)
    else:
        tmp.append(3)
    # activation type, TODO: check attr name
    tmp.append('None')
    # batch size
    tmp.append(1)
    # input channels, height, width 
    in_shapes = blocks.vars[current_op.input('X')[0]].shape
    for i in range(1, len(in_shapes)):
        tmp.append(int(in_shapes[i]))
    return tmp


def activation_op_params(blocks, current_op):
    """Getting params of activation op
    Args:
        blocks, current block
        current_op, current op
    Returns:
        list, op name and hyperparamters
    """
    tmp = []
    # op name
    tmp.append('activation')
    # cluster, threads, test_iter
    tmp = tmp + [0, 1, 100]
    # activation type
    tmp.append(current_op.type)
    # batch size
    tmp.append(1)
    # input channels, height, width
    in_shapes = blocks.vars[current_op.input('X')[0]].shape
    for i in range(1, len(in_shapes)):
        tmp.append(int(in_shapes[i]))
    return tmp


def pooling_op_params(blocks, current_op):
    """Getting params of pooling op
    Args:
        blocks, current block
        current_op, current op
    Returns:
        list, op name and hyperparamters
    """
    tmp, res = [], []
    # op name
    tmp.append('pooling')
    # cluster, threads, test_iters
    tmp = tmp + [0, 1, 100]
    # global pooling
    tmp.append(int(current_op.attr('global_pooling')))
    # batch size
    tmp.append(1)
    # channels, height, width
    in_shapes = blocks.vars[current_op.input('X')[0]].shape
    tmp = tmp + [int(in_shapes[1]), int(in_shapes[2]), int(in_shapes[3])]
    # kernel size
    ksize = current_op.attr('ksize')
    tmp.append(int(ksize[0]))
    if ksize[0] != ksize[1]:
        res.append(int(ksize[1]))

    # padding
    paddings = current_op.attr('paddings')
    tmp.append(int(paddings[0]))
    if paddings[0] != paddings[1]:
        res.append(int(paddings[1]))

    # stride
    strides = current_op.attr('strides')
    tmp.append(int(strides[0]))
    if strides[0] != strides[1]:
        res.append(int(strides[1]))

    # ceil mode
    tmp.append(int(current_op.attr('ceil_mode')))

    # pool type
    pool_type = current_op.attr('pooling_type')
    exclusive = current_op.attr('exclusive')
    if pool_type == 'max' and (not exclusive):
        tmp.append(1)
    elif pool_type == 'avg' and (not exclusive):
        tmp.append(2)
    else:
        tmp.append(3)

    tmp = tmp + res
    return tmp


def softmax_op_params(blocks, current_op):
    #TODO: Getting params of softmax op
    return []


def resize_op_params(blocks, current_op):
    #TODO: Getting params of resize op
    return []


def fc_op_params(blocks, current_op, isbias):
    """Getting params of fc op
    Note: 
        fc op is converted to conv op with 1x1 kernels
    Args:
        blocks, current block
        current_op, current op
    Returns:
        list, op name and hyperparamters
    """
    # op name, clusters, threads, test_iters
    tmp = ['conv', 0, 1, 100]
    # flag bias
    tmp.append(int(isbias))
    # flag relu
    tmp.append(0)
    # batch size 
    tmp.append(1)
    # input channels, height, width
    channels = blocks.vars[current_op.input('X')[0]].shape[1]
    tmp = tmp + [int(channels), 1, 1]
    # output channels
    tmp.append(int(blocks.vars[current_op.output('Out')[0]].shape[1]))
    # groups, kernel size, padding, stride, dilation
    tmp = tmp + [1, 1, 0, 1, 1]
    return tmp


def write_lookup_table(params, output_file):
    """Writing down all params to a txt file
    Args:
        params, list of op params
        output_file, string of output file path
    Returns:
        None
    """
    fw = open(output_file, 'w')
    for line in params:
        str_list = []
        for item in line:
            str_list.append(str(item))
        fw.write(' '.join(str_list) + '\n')
    fw.close()


def get_ops_from_program(program, output_file=None):
    """Getting ops params from a paddle program
    Args:
        program, fluid program desc
        output_file, string of output file path
    Returns:
        list, params
    """
    blocks = program.global_block()
    params = []
    i = 0
    while i < len(blocks.ops):
        current_op = blocks.ops[i]
        if current_op.type in ['conv2d', 'depthwise_conv2d']:
            tmp = conv_op_params(blocks, current_op)
        elif current_op.type in [
                'elementwise_add', 'elementwise_mul', 'elementwise_max'
        ]:
            tmp = eltwise_op_params(blocks, current_op)
        elif current_op.type in [
                'relu', 'prelu', 'sigmoid', 'relu6', 'elu', 'brelu',
                'leaky_relu'
        ]:
            tmp = activation_op_params(blocks, current_op)
        elif current_op.type == 'pool2d':
            tmp = pooling_op_params(blocks, current_op)
        elif current_op.type == 'softmax':
            tmp = softmax_op_params(blocks, current_op)
        elif current_op.type == 'image_resize':
            tmp = resize_op_params(blocks, current_op)
        elif current_op.type == 'mul':
            try:
                next_op = blocks.ops[i + 1]
                if next_op.type == 'elementwise_add':
                    tmp = fc_op_params(blocks, current_op, True)
                    i = i + 1
                else:
                    tmp = fc_op_params(blocks, current_op, False)
            except:
                tmp = fc_op_params(blocks, current_op, False)
        else:
            tmp = []
            #print('{} op is not support right now...'.format(current_op.type))

        if len(tmp) > 0:
            params.append(tuple(tmp))

        i += 1

    if output_file is not None:
        write_lookup_table(params, output_file)

    return params
