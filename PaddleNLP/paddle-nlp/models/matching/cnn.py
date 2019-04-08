"""
cnn class
"""

import paddle_layers as layers


class CNN(object):
    """
    CNN
    """

    def __init__(self, conf_dict):
        """
        initialize
        """
        self.dict_size = conf_dict["dict_size"]
        self.task_mode = conf_dict["task_mode"]
        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.filter_size = conf_dict["net"]["filter_size"]
        self.num_filters = conf_dict["net"]["num_filters"]
        self.hidden_dim = conf_dict["net"]["hidden_dim"]

    def predict(self, left, right):
        """
        Forward network
        """
        # embedding layer
        emb_layer = layers.EmbeddingLayer(self.dict_size, self.emb_dim, "emb")
        left_emb = emb_layer.ops(left)
        right_emb = emb_layer.ops(right)
        # Presentation context
        cnn_layer = layers.SequenceConvPoolLayer(
            self.filter_size, self.num_filters, "conv")
        left_cnn = cnn_layer.ops(left_emb)
        right_cnn = cnn_layer.ops(right_emb)
        # matching layer
        if self.task_mode == "pairwise":
            relu_layer = layers.FCLayer(self.hidden_dim, "relu", "relu")
            left_relu = relu_layer.ops(left_cnn)
            right_relu = relu_layer.ops(right_cnn)
            cos_sim_layer = layers.CosSimLayer()
            pred = cos_sim_layer.ops(left_relu, right_relu)
            return left_relu, pred
        else:
            concat_layer = layers.ConcatLayer(1)
            concat = concat_layer.ops([left_cnn, right_cnn])
            relu_layer = layers.FCLayer(self.hidden_dim, "relu", "relu")
            concat_fc = relu_layer.ops(concat)
            softmax_layer = layers.FCLayer(2, "softmax", "cos_sim")
            pred = softmax_layer.ops(concat_fc)
            return left_cnn, pred
