from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from absl import flags
from paddle import fluid

import build.layers as layers
import build.ops as _ops

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_stages", 3, "number of stages")
flags.DEFINE_integer("num_cells", 7, "number of cells per stage")
flags.DEFINE_integer("width", 96, "network width")

flags.DEFINE_integer("ratio", 6, "compression ratio")

flags.DEFINE_float("dropout_rate_path", 0.4, "dropout rate for cell path")
flags.DEFINE_float("dropout_rate_fin", 0.5, "dropout rate for finishing layer")

num_classes = 10

ops = [
    _ops.conv_1x1,
    _ops.conv_3x3,
    _ops.conv_5x5,
    _ops.dilated_3x3,
    _ops.conv_1x3_3x1,
    _ops.conv_1x5_5x1,
    _ops.maxpool_3x3,
    _ops.maxpool_5x5,
    _ops.avgpool_3x3,
    _ops.avgpool_5x5,
]


def net(inputs, output, tokens):
    adjvec = tokens[1]
    tokens = tokens[0]
    print("tokens: " + str(tokens))
    print("adjvec: " + str(adjvec))

    num_nodes = len(tokens) // 2

    def slice(vec):
        mat = np.zeros([num_nodes, num_nodes])
        pos = lambda i: i * (i - 1) // 2
        for i in range(1, num_nodes):
            mat[0:i, i] = vec[pos(i):pos(i + 1)]
        return mat

    normal_to, reduce_to = np.split(tokens, 2)
    normal_ad, reduce_ad = map(slice, np.split(adjvec, 2))

    # with tf.variable_scope("0.initial_conv"):
    x = layers.conv(inputs, FLAGS.width, (3, 3))
    pre_activation_idx = [1]
    reduction_idx = [
        i * FLAGS.num_cells + 1 for i in range(1, FLAGS.num_stages)
    ]
    aux_head_idx = [(FLAGS.num_stages - 1) * FLAGS.num_cells]

    num_cells = FLAGS.num_stages * FLAGS.num_cells
    for c in range(1, num_cells + 1):
        dropout_rate = c / num_cells * FLAGS.dropout_rate_path
        if c in pre_activation_idx:
            # with tf.variable_scope("%d.normal_cell" % c):
            x = cell(x, normal_to, normal_ad, dropout_rate, pre_activation=True)
        elif c in reduction_idx:
            # with tf.variable_scope("%d.reduction_cell" % c):
            x = cell(x, reduce_to, reduce_ad, dropout_rate, downsample=True)
        else:
            # with tf.variable_scope("%d.normal_cell" % c):
            x = cell(x, normal_to, normal_ad, dropout_rate)
        if c in aux_head_idx:
            # with tf.variable_scope("aux_head"):
            aux_loss = aux_head(x, output)

    # with tf.variable_scope("%d.global_average_pooling" % (num_cells + 1)):
    print("main:" + str(x.shape))
    x = layers.bn_relu(x)
    x = layers.global_avgpool(x)

    x = layers.dropout(x, dropout_rate)
    logits = layers.fully_connected(x, num_classes)

    cost = fluid.layers.softmax_with_cross_entropy(logits=logits, label=output)
    avg_cost = fluid.layers.mean(cost) + 0.4 * aux_loss
    accuracy = fluid.layers.accuracy(input=logits, label=output)

    return avg_cost, accuracy


def aux_head(inputs, output):
    print("aux_input: " + str(inputs.shape))

    # x = layers.avgpool(inputs, (5, 5), (3, 3), padding="valid")
    x = layers.avgpool_valid(inputs, (5, 5), (3, 3))
    print("aux:" + str(x.shape))

    x = layers.conv(x, 128, (1, 1))
    print("aux:" + str(x.shape))
    x = layers.bn_relu(x)
    print("aux:" + str(x.shape))
    # x = layers.conv(x, 768, (4, 4), padding="valid")
    x = layers.conv(x, 768, (4, 4), auto_pad=False)
    print("aux:" + str(x.shape))

    # x = tf.squeeze(x, axis=[2, 3])
    x = layers.bn_relu(x)
    logits = layers.fully_connected(x, num_classes)

    cost = fluid.layers.softmax_with_cross_entropy(logits=logits, label=output)
    return fluid.layers.mean(cost)


def cell(inputs,
         tokens,
         adjmat,
         dropout_rate,
         pre_activation=False,
         downsample=False):
    filters = int(inputs.shape[1])
    d = filters // FLAGS.ratio

    if pre_activation:
        inputs = layers.bn_relu(inputs)

    num_nodes, tensors = len(adjmat), []
    for n in range(num_nodes):
        func = ops[tokens[n]]
        idx, = np.nonzero(adjmat[:, n])
        # with tf.variable_scope("%d.%s" % (n, func.__name__)):
        if len(idx) == 0:
            x = inputs if pre_activation else layers.bn_relu(inputs)

            x = layers.conv(x, d, (1, 1))
            x = layers.bn_relu(x)
            x = func(x, downsample)
        else:
            # x = tf.add_n([tensors[i] for i in idx])
            tensor_list = [tensors[i] for i in idx]
            x = tensor_list[0]
            for i in range(1, len(tensor_list)):
                x = fluid.layers.elementwise_add(x, tensor_list[i])

            x = layers.bn_relu(x)
            x = func(x)
        x = layers.dropout(x, dropout_rate)
        tensors.append(x)

    free_ends, = np.where(~adjmat.any(axis=1))
    tensors = [tensors[i] for i in free_ends]
    filters = filters * 2 if downsample else filters
    # with tf.variable_scope("%d.add" % num_nodes):
    #     x = tf.concat(tensors, axis=1)
    x = fluid.layers.concat(tensors, axis=1)
    x = layers.conv(x, filters, (1, 1))

    print("cell: %s -> %s" % (inputs.shape, x.shape))
    return x
