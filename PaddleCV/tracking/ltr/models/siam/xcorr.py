import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn

from pytracking.libs.Fconv2d import FConv2D


def xcorr(x, kernel):
    """group conv2d to calculate cross correlation
    """
    batch = kernel.shape[0]
    px = fluid.layers.reshape(x, [1, -1, x.shape[2], x.shape[3]])
    pk = fluid.layers.reshape(kernel, [-1, x.shape[1], kernel.shape[2], kernel.shape[3]])
    scores_map = FConv2D(px, pk, stride=1, padding=0, dilation=1, groups=batch)
    scores_map = fluid.layers.reshape(
        scores_map, [batch, -1, scores_map.shape[2], scores_map.shape[3]])
    return scores_map


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.shape[0]
    channel = kernel.shape[1]
    px = fluid.layers.reshape(x, [1, -1, x.shape[2], x.shape[3]])
    pk = fluid.layers.reshape(kernel, [-1, 1, kernel.shape[2], kernel.shape[3]])
    scores_map = FConv2D(px, pk, stride=1, padding=0, dilation=1, groups=batch*channel)
    scores_map = fluid.layers.reshape(
        scores_map,[batch, -1, scores_map.shape[2], scores_map.shape[3]])
    return scores_map
