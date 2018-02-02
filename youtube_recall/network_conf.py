#!/usr/bin/env python
# -*- coding: utf-8 -*-
import paddle.v2 as paddle
import cPickle


class DNNmodel(object):
    """
    Deep Neural Networks for YouTube candidate generation
    """

    def __init__(self,
                 dnn_layer_dims=None,
                 feature_dict=None,
                 item_freq=None,
                 is_infer=False):
        """
        initialize model
        @dnn_layer_dims: dimension of each hidden layer
        @feature_dict: dictionary of encoded feature
        @item_freq: dictionary of feature values and its frequency
        @is_infer: if infer mode
        """
        self._dnn_layer_dims = dnn_layer_dims
        self._feature_dict = feature_dict
        self._item_freq = item_freq

        self._is_infer = is_infer

        # build model
        self._build_input_layer()
        self._build_embedding_layer()
        self.model_cost = self._build_dnn_model()

    def _build_input_layer(self):
        """
        build input layer
        """
        self._history_clicked_items = paddle.layer.data(
            name="history_clicked_items",
            type=paddle.data_type.integer_value_sequence(
                len(self._feature_dict['history_clicked_items'])))
        self._history_clicked_categories = paddle.layer.data(
            name="history_clicked_categories",
            type=paddle.data_type.integer_value_sequence(
                len(self._feature_dict['history_clicked_categories'])))
        self._history_clicked_tags = paddle.layer.data(
            name="history_clicked_tags",
            type=paddle.data_type.integer_value_sequence(
                len(self._feature_dict['history_clicked_tags'])))
        self._user_id = paddle.layer.data(
            name="user_id",
            type=paddle.data_type.integer_value(
                len(self._feature_dict['user_id'])))
        self._province = paddle.layer.data(
            name="province",
            type=paddle.data_type.integer_value(
                len(self._feature_dict['province'])))
        self._city = paddle.layer.data(
            name="city",
            type=paddle.data_type.integer_value(
                len(self._feature_dict['city'])))
        self._phone = paddle.layer.data(
            name="phone",
            type=paddle.data_type.integer_value(
                len(self._feature_dict['phone'])))
        self._target_item = paddle.layer.data(
            name="target_item",
            type=paddle.data_type.integer_value(
                len(self._feature_dict['history_clicked_items'])))

    def _create_emb_attr(self, name):
        """
        create embedding parameter
        """
        return paddle.attr.Param(
            name=name,
            initial_std=0.001,
            learning_rate=1,
            l2_rate=0,
            sparse_update=False)

    def _build_embedding_layer(self):
        """
        build embedding layer
        """
        self._user_id_emb = paddle.layer.embedding(
            input=self._user_id,
            size=64,
            param_attr=self._create_emb_attr('_proj_user_id'))
        self._province_emb = paddle.layer.embedding(
            input=self._province,
            size=8,
            param_attr=self._create_emb_attr('_proj_province'))
        self._city_emb = paddle.layer.embedding(
            input=self._city,
            size=16,
            param_attr=self._create_emb_attr('_proj_city'))
        self._phone_emb = paddle.layer.embedding(
            input=self._phone,
            size=16,
            param_attr=self._create_emb_attr('_proj_phone'))
        self._history_clicked_items_emb = paddle.layer.embedding(
            input=self._history_clicked_items,
            size=64,
            param_attr=self._create_emb_attr('_proj_history_clicked_items'))
        self._history_clicked_categories_emb = paddle.layer.embedding(
            input=self._history_clicked_categories,
            size=8,
            param_attr=self._create_emb_attr(
                '_proj_history_clicked_categories'))
        self._history_clicked_tags_emb = paddle.layer.embedding(
            input=self._history_clicked_tags,
            size=64,
            param_attr=self._create_emb_attr('_proj_history_clicked_tags'))

    def _build_dnn_model(self):
        """
        build dnn model
        """
        self._rnn_cell = paddle.networks.simple_lstm(
            input=self._history_clicked_items_emb, size=64)
        self._lstm_last = paddle.layer.pooling(
            input=self._rnn_cell, pooling_type=paddle.pooling.Max())
        self._avg_emb_cats = paddle.layer.pooling(
            input=self._history_clicked_categories_emb,
            pooling_type=paddle.pooling.Avg())
        self._avg_emb_tags = paddle.layer.pooling(
            input=self._history_clicked_tags_emb,
            pooling_type=paddle.pooling.Avg())
        self._fc_0 = paddle.layer.fc(
            name="Relu1",
            input=[
                self._lstm_last, self._user_id_emb, self._province_emb,
                self._city_emb, self._avg_emb_cats, self._avg_emb_tags,
                self._phone_emb
            ],
            size=self._dnn_layer_dims[0],
            act=paddle.activation.Relu())

        self._fc_1 = paddle.layer.fc(name="Relu2",
                                     input=self._fc_0,
                                     size=self._dnn_layer_dims[1],
                                     act=paddle.activation.Relu())

        if not self._is_infer:
            return paddle.layer.nce(
                input=self._fc_1,
                label=self._target_item,
                num_classes=len(self._feature_dict['history_clicked_items']),
                param_attr=paddle.attr.Param(name="nce_w"),
                bias_attr=paddle.attr.Param(name="nce_b"),
                num_neg_samples=5,
                neg_distribution=self._item_freq)
        else:
            self.prediction_layer = paddle.layer.mixed(
                size=len(self._feature_dict['history_clicked_items']),
                input=paddle.layer.trans_full_matrix_projection(
                    self._fc_1, param_attr=paddle.attr.Param(name="nce_w")),
                act=paddle.activation.Softmax(),
                bias_attr=paddle.attr.Param(name="nce_b"))
            return self.prediction_layer, self._fc_1


if __name__ == "__main__":
    # this is to test and debug the network topology defination.
    # please set the hyper-parameters as needed.
    item_freq_path = "./output/item_freq.pkl"
    with open(item_freq_path) as f:
        item_freq = cPickle.load(f)

    feature_dict_path = "./output/feature_dict.pkl"
    with open(feature_dict_path) as f:
        feature_dict = cPickle.load(f)

    a = DNNmodel(
        dnn_layer_dims=[256, 31],
        feature_dict=feature_dict,
        item_freq=item_freq,
        is_infer=False)
