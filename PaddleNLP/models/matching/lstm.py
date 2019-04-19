"""
lstm class
"""
import paddle_layers as layers


class LSTM(object):
    """
    LSTM
    """

    def __init__(self, conf_dict):
        """
        initialize
        """
        self.dict_size = conf_dict["dict_size"]
        self.task_mode = conf_dict["task_mode"]
        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.lstm_dim = conf_dict["net"]["lstm_dim"]
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
        lstm_layer = layers.DynamicLSTMLayer(self.lstm_dim, "lstm")
        left_lstm = lstm_layer.ops(left_emb)
        right_lstm = lstm_layer.ops(right_emb)
        last_layer = layers.SequenceLastStepLayer()
        left_last = last_layer.ops(left_lstm)
        right_last = last_layer.ops(right_lstm)
        # matching layer
        if self.task_mode == "pairwise":
            relu_layer = layers.FCLayer(self.hidden_dim, "relu", "relu")
            left_relu = relu_layer.ops(left_last)
            right_relu = relu_layer.ops(right_last)
            cos_sim_layer = layers.CosSimLayer()
            pred = cos_sim_layer.ops(left_relu, right_relu)
            return left_relu, pred
        else:
            concat_layer = layers.ConcatLayer(1)
            concat = concat_layer.ops([left_last, right_last])
            relu_layer = layers.FCLayer(self.hidden_dim, "relu", "relu")
            concat_fc = relu_layer.ops(concat)
            softmax_layer = layers.FCLayer(2, "softmax", "cos_sim")
            pred = softmax_layer.ops(concat_fc)
            return left_last, pred
