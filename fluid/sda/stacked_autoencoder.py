import numpy as np
import pdb

import paddle.v2 as paddle
import paddle.fluid as fluid

from utils import cross_entropy_loss
from utils import get_da_weight


class stacked_denoise_autoencoder():
    def __init__(self, args):
        self.layers = []
        self.num_layers = args.num_layers
        self.class_num = args.class_num
        self.weights = {}

    def build_model(self, layer_input):
        for i in range(len(self.num_layers)):
            sigmoid_layer = fluid.layers.fc(
                input=layer_input,
                size=self.num_layers[i],
                act='sigmoid',
                param_attr=fluid.ParamAttr(
                    name='s%d_w' % i, initializer=fluid.initializer.Normal()),
                bias_attr=fluid.ParamAttr(
                    name='s%d_b' % i, initializer=fluid.initializer.Constant()))
            layer_input = sigmoid_layer

        out = fluid.layers.fc(input=sigmoid_layer,
                              size=self.class_num,
                              act='softmax',
                              param_attr=fluid.ParamAttr(
                                  name='out_w',
                                  initializer=fluid.initializer.Normal()),
                              bias_attr=fluid.ParamAttr(
                                  name='out_b',
                                  initializer=fluid.initializer.Constant()))
        return out

    def da_model(self, layer_input, n_visible, n_hidden, num_layer):
        W = fluid.layers.create_parameter(
            shape=[n_visible, n_hidden],
            dtype='float32',
            attr=fluid.ParamAttr(
                name='s%d_w' % num_layer,
                initializer=fluid.initializer.Normal()),
            is_bias=False)
        bvis = fluid.layers.create_parameter(
            shape=[n_visible],
            dtype='float32',
            attr=fluid.ParamAttr(
                name='s%d_b_out' % num_layer,
                initializer=fluid.initializer.Constant()),
            is_bias=True)
        bhid = fluid.layers.create_parameter(
            shape=[n_hidden],
            dtype='float32',
            attr=fluid.ParamAttr(
                name='s%d_b' % num_layer,
                initializer=fluid.initializer.Constant()),
            is_bias=True)
        hidden_value = fluid.layers.sigmoid(
            fluid.layers.matmul(layer_input, W) + bhid)
        predict = fluid.layers.sigmoid(
            fluid.layers.matmul(
                hidden_value, W, transpose_y=True) + bvis)
        return predict, hidden_value
