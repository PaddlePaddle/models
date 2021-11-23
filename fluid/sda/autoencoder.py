import numpy as np
import paddle.v2 as paddle
import paddle.fluid as fluid


def denoise_autoencoder(input, args):
    n_hidden = args.n_hidden
    n_visible = args.img_height * args.img_width
    W = fluid.layers.create_parameter(
        shape=[n_visible, n_hidden],
        dtype='float32',
        attr=fluid.ParamAttr(
            name='W', initializer=fluid.initializer.Normal()),
        is_bias=False)
    bvis = fluid.layers.zeros(shape=[n_visible], dtype='float32')
    bhid = fluid.layers.zeros(shape=[n_hidden], dtype='float32')
    hidden_value = fluid.layers.sigmoid(fluid.layers.matmul(input, W) + bhid)
    out = fluid.layers.sigmoid(
        fluid.layers.matmul(
            hidden_value, W, transpose_y=True) + bvis)
    return out
