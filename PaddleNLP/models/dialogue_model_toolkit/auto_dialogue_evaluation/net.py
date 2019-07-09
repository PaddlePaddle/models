"""
Network for auto dialogue evaluation
"""
import os
import sys
import time
import six

import numpy as np
import math
import paddle.fluid as fluid
import paddle


class Network(object):
    """
    Network
    """

    def __init__(self,
                 vocab_size,
                 emb_size,
                 hidden_size,
                 clip_value=10.0,
                 word_emb_name="shared_word_emb",
                 lstm_W_name="shared_lstm_W",
                 lstm_bias_name="shared_lstm_bias"):
        """
        Init function
        """

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.clip_value = clip_value
        self.word_emb_name = word_emb_name
        self.lstm_W_name = lstm_W_name
        self.lstm_bias_name = lstm_bias_name

    def network(self, loss_type='CLS'):
        """
        Network definition 
        """
        #Input data
        context_wordseq = fluid.layers.data(
            name="context_wordseq", shape=[1], dtype="int64", lod_level=1)
        response_wordseq = fluid.layers.data(
            name="response_wordseq", shape=[1], dtype="int64", lod_level=1)
        label = fluid.layers.data(name="label", shape=[1], dtype="float32")

        self._feed_name = ["context_wordseq", "response_wordseq", "label"]
        self._feed_infer_name = ["context_wordseq", "response_wordseq"]

        #emb
        context_emb = fluid.layers.embedding(
            input=context_wordseq,
            size=[self.vocab_size, self.emb_size],
            is_sparse=True,
            param_attr=fluid.ParamAttr(
                name=self.word_emb_name,
                initializer=fluid.initializer.Normal(scale=0.1)))

        response_emb = fluid.layers.embedding(
            input=response_wordseq,
            size=[self.vocab_size, self.emb_size],
            is_sparse=True,
            param_attr=fluid.ParamAttr(
                name=self.word_emb_name,
                initializer=fluid.initializer.Normal(scale=0.1)))

        #fc to fit dynamic LSTM
        context_fc = fluid.layers.fc(
            input=context_emb,
            size=self.hidden_size * 4,
            param_attr=fluid.ParamAttr(name='fc_weight'),
            bias_attr=fluid.ParamAttr(name='fc_bias'))

        response_fc = fluid.layers.fc(
            input=response_emb,
            size=self.hidden_size * 4,
            param_attr=fluid.ParamAttr(name='fc_weight'),
            bias_attr=fluid.ParamAttr(name='fc_bias'))

        #LSTM
        context_rep, _ = fluid.layers.dynamic_lstm(
            input=context_fc,
            size=self.hidden_size * 4,
            param_attr=fluid.ParamAttr(name=self.lstm_W_name),
            bias_attr=fluid.ParamAttr(name=self.lstm_bias_name))
        context_rep = fluid.layers.sequence_last_step(context_rep)
        print('context_rep shape: %s' % str(context_rep.shape))

        response_rep, _ = fluid.layers.dynamic_lstm(
            input=response_fc,
            size=self.hidden_size * 4,
            param_attr=fluid.ParamAttr(name=self.lstm_W_name),
            bias_attr=fluid.ParamAttr(name=self.lstm_bias_name))
        response_rep = fluid.layers.sequence_last_step(input=response_rep)
        print('response_rep shape: %s' % str(response_rep.shape))

        logits = fluid.layers.bilinear_tensor_product(
            context_rep, response_rep, size=1)
        print('logits shape: %s' % str(logits.shape))  #[batch,1]

        if loss_type == 'CLS':
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
            print('before reduce mean loss shape: %s' % str(loss.shape))
            loss = fluid.layers.reduce_mean(
                fluid.layers.clip(
                    loss, min=-self.clip_value, max=self.clip_value))
            print('after reduce mean loss shape: %s' % str(loss.shape))
        elif loss_type == 'L2':
            norm_score = 2 * fluid.layers.sigmoid(logits)
            loss = fluid.layers.square_error_cost(norm_score, label) / 4
            loss = fluid.layers.reduce_mean(loss)
        else:
            raise ValueError

        return logits, loss

    def set_word_embedding(self, word_emb, place):
        """
        Set word embedding
        """
        word_emb_param = fluid.global_scope().find_var(
            self.word_emb_name).get_tensor()
        word_emb_param.set(word_emb, place)

    def get_feed_names(self):
        """
        Return feed names
        """
        return self._feed_name

    def get_feed_inference_names(self):
        """
        Return inference names
        """
        return self._feed_infer_name
