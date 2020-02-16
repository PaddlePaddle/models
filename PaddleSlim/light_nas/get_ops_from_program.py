# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Get ops from program."""


def conv_op_params(blocks, current_op):
    """Getting params of conv op
    Args:
        blocks: BlockDesc, current block
        current_op: OpDesc, current op
    Returns:
        (list): op name and hyperparamters
    """
    tmp, res = [], []
    # op_name
    tmp.append('conv')
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
        blocks: BlockDesc, current block
        current_op: OpDesc, current op
    Returns:
        (list): op name and hyperparamters
    """
    tmp = []
    # op name
    tmp.append('batch_norm')
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
        blocks: BlockDesc, current block
        current_op: OpDesc, current op
    Returns:
        (list): op name and hyperparamters
    """
    # op name
    tmp = ['eltwise']
    # elementwise type, TODO: add more ops
    if current_op.type == 'elementwise_mul':
        tmp.append(1)
    elif current_op.type == 'elementwise_add':
        tmp.append(2)
    else:
        tmp.append(3)
    # batch size
    tmp.append(1)
    # input channels, height, width 
    in_shapes = blocks.vars[current_op.input('X')[0]].shape
    while len(in_shapes) < 4:
        in_shapes = in_shapes + (1, )

    for i in range(1, len(in_shapes)):
        tmp.append(int(in_shapes[i]))
    return tmp


def activation_op_params(blocks, current_op):
    """Getting params of activation op
    Args:
        blocks: BlockDesc, current block
        current_op: OpDesc, current op
    Returns:
        (list): op name and hyperparamters
    """
    tmp = []
    # op name
    tmp.append('activation')
    # activation type
    tmp.append(current_op.type)
    # batch size
    tmp.append(1)
    # input channels, height, width
    in_shapes = blocks.vars[current_op.input('X')[0]].shape
    while len(in_shapes) < 4:
        in_shapes = in_shapes + (1, )

    for i in range(1, len(in_shapes)):
        tmp.append(int(in_shapes[i]))
    return tmp


def pooling_op_params(blocks, current_op):
    """Getting params of pooling op
    Args:
        blocks: BlockDesc, current block
        current_op: OpDesc, current op
    Returns:
        (list): op name and hyperparamters    
    """
    tmp, res = [], []
    # op name
    tmp.append('pooling')
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
    """Getting params of softmax op
    Args:
        blocks: BlockDesc, current block
        current_op: OpDesc, current op
    Returns:
        (list): op name and hyperparamters
    """
    # op name
    tmp = ['softmax']
    # axis
    tmp.append(current_op.attr('axis'))
    # batch size
    tmp.append(1)
    # input channels, height, width
    in_shapes = blocks.vars[current_op.input('X')[0]].shape
    while len(in_shapes) < 4:
        in_shapes = in_shapes + (1, )

    for i in range(1, len(in_shapes)):
        tmp.append(int(in_shapes[i]))

    return tmp


def fc_op_params(blocks, current_op):
    """Getting params of fc op
    Note: 
        fc op is converted to conv op with 1x1 kernels
    Args:
        blocks: BlockDesc, current block
        current_op: OpDesc, current op
    Returns:
        (list): op name and hyperparamters
    """
    # op name
    tmp = ['conv']
    # flag bias
    tmp.append(0)
    # flag relu
    tmp.append(0)
    # batch size 
    tmp.append(1)
    # input channels, height, width
    channels = 1
    in_shape = blocks.vars[current_op.input('X')[0]].shape
    for i in range(1, len(in_shape)):
        channels *= in_shape[i]
    tmp = tmp + [int(channels), 1, 1]
    # output channels
    tmp.append(int(blocks.vars[current_op.output('Out')[0]].shape[1]))
    # groups, kernel size, padding, stride, dilation
    tmp = tmp + [1, 1, 0, 1, 1]
    return tmp


def get_ops_from_program(program):
    """Getting ops params from a paddle program
    Args:
        program(Program): The program to get ops.
    Returns:
        (list): ops.
    """
    blocks = program.global_block()
    ops = []
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
        elif current_op.type == 'batch_norm':
            tmp = batch_norm_op_params(blocks, current_op)
        elif current_op.type == 'pool2d':
            tmp = pooling_op_params(blocks, current_op)
        elif current_op.type == 'batch_norm':
            tmp = batch_norm_op_params(blocks, current_op)
        elif current_op.type == 'softmax':
            tmp = softmax_op_params(blocks, current_op)
        elif current_op.type == 'mul':
            tmp = fc_op_params(blocks, current_op)
        else:
            tmp = None
        if tmp:
            ops.append(tmp)
        i += 1
    return ops
