import paddle.fluid as fluid
from paddle.fluid import ParamAttr
import numpy as np


class LSTMAttentionModel(object):
    """LSTM Attention Model"""

    def __init__(self,
                 bias_attr,
                 embedding_size=512,
                 lstm_size=1024,
                 drop_rate=0.5):
        self.lstm_size = lstm_size
        self.embedding_size = embedding_size
        self.bias_attr = ParamAttr(
            regularizer=fluid.regularizer.L2Decay(0.0),
            initializer=fluid.initializer.NormalInitializer(scale=0.0))

    def forward(self, input, is_training):
        input_fc = fluid.layers.fc(
            input=input,
            size=self.embedding_size,
            act='tanh',
            bias_attr=ParamAttr(
                regularizer=fluid.regularizer.L2Decay(0.0),
                initializer=fluid.initializer.NormalInitializer(scale=0.0)))

        lstm_forward_fc = fluid.layers.fc(
            input=input_fc,
            size=self.lstm_size * 4,
            act=None,
            bias_attr=ParamAttr(
                regularizer=fluid.regularizer.L2Decay(0.0),
                initializer=fluid.initializer.NormalInitializer(scale=0.0)))
        lstm_forward, _ = fluid.layers.dynamic_lstm(
            input=lstm_forward_fc, size=self.lstm_size * 4, is_reverse=False)

        lsmt_backward_fc = fluid.layers.fc(
            input=input_fc,
            size=self.lstm_size * 4,
            act=None,
            bias_attr=ParamAttr(
                regularizer=fluid.regularizer.L2Decay(0.0),
                initializer=fluid.initializer.NormalInitializer(scale=0.0)))
        lstm_backward, _ = fluid.layers.dynamic_lstm(
            input=lsmt_backward_fc, size=self.lstm_size * 4, is_reverse=True)

        lstm_concat = fluid.layers.concat(
            input=[lstm_forward, lstm_backward], axis=1)

        lstm_dropout = fluid.layers.dropout(
            x=lstm_concat, dropout_prob=0.5, is_test=(not is_training))

        lstm_weight = fluid.layers.fc(
            input=lstm_dropout,
            size=1,
            act='sequence_softmax',
            bias_attr=ParamAttr(
                regularizer=fluid.regularizer.L2Decay(0.0),
                initializer=fluid.initializer.NormalInitializer(scale=0.0)))
        scaled = fluid.layers.elementwise_mul(
            x=lstm_dropout, y=lstm_weight, axis=0)
        lstm_pool = fluid.layers.sequence_pool(input=scaled, pool_type='sum')

        return lstm_pool
