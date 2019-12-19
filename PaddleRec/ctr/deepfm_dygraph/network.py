import math

import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import FC, Embedding


class DeepFM(fluid.dygraph.Layer):
    def __init__(self, scope_name, args):
        super(DeepFM, self).__init__(scope_name)
        self.args = args
        self.init_value_ = 0.1

        self.fm = FM(self.full_name(), args)
        self.dnn = DNN(self.full_name(), args)

    def forward(self, raw_feat_idx, raw_feat_value, label):
        feat_idx = fluid.layers.reshape(raw_feat_idx,
                                        [-1, 1])  # (None * num_field) * 1
        feat_value = fluid.layers.reshape(
            raw_feat_value,
            [-1, self.args.num_field, 1])  # None * num_field * 1

        y_first_order, y_second_order, feat_embeddings = self.fm(feat_idx,
                                                                 feat_value)
        y_dnn = self.dnn(feat_embeddings)

        predict = fluid.layers.sigmoid(y_first_order + y_second_order + y_dnn)

        return predict


class FM(fluid.dygraph.Layer):
    def __init__(self, scope_name, args):
        super(FM, self).__init__(scope_name)
        self.args = args
        self.init_value_ = 0.1
        self.embedding_w = Embedding(
            self.full_name(),
            is_sparse=self.args.is_sparse,
            dtype='float32',
            size=[self.args.num_feat + 1, 1],
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=self.init_value_),
                regularizer=fluid.regularizer.L1DecayRegularizer(
                    self.args.reg)))
        self.embedding = Embedding(
            self.full_name(),
            is_sparse=self.args.is_sparse,
            dtype='float32',
            size=[self.args.num_feat + 1, self.args.embedding_size],
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0,
                    scale=self.init_value_ /
                    math.sqrt(float(self.args.embedding_size)))))

    def forward(self, feat_idx, feat_value):
        # -------------------- first order term  --------------------
        first_weights_re = self.embedding_w(feat_idx)
        first_weights = fluid.layers.reshape(
            first_weights_re,
            shape=[-1, self.args.num_field, 1])  # None * num_field * 1
        y_first_order = fluid.layers.reduce_sum((first_weights * feat_value), 1)

        # -------------------- second order term  --------------------
        feat_embeddings_re = self.embedding(feat_idx)
        feat_embeddings = fluid.layers.reshape(
            feat_embeddings_re,
            shape=[-1, self.args.num_field, self.args.embedding_size
                   ])  # None * num_field * embedding_size
        feat_embeddings = feat_embeddings * feat_value  # None * num_field * embedding_size

        # sum_square part
        summed_features_emb = fluid.layers.reduce_sum(
            feat_embeddings, 1)  # None * embedding_size
        summed_features_emb_square = fluid.layers.square(
            summed_features_emb)  # None * embedding_size

        # square_sum part
        squared_features_emb = fluid.layers.square(
            feat_embeddings)  # None * num_field * embedding_size
        squared_sum_features_emb = fluid.layers.reduce_sum(
            squared_features_emb, 1)  # None * embedding_size

        y_second_order = 0.5 * fluid.layers.reduce_sum(
            summed_features_emb_square - squared_sum_features_emb,
            1,
            keep_dim=True)  # None * 1

        return y_first_order, y_second_order, feat_embeddings


class DNN(fluid.dygraph.Layer):
    def __init__(self, scope_name, args):
        super(DNN, self).__init__(scope_name)
        self.args = args
        self.init_value_ = 0.1
        sizes = self.args.layer_sizes + [1]
        acts = [self.args.act
                for _ in range(len(self.args.layer_sizes))] + [None]
        w_scales = [
            self.init_value_ / math.sqrt(float(10))
            for _ in range(len(self.args.layer_sizes))
        ] + [self.init_value_]
        self.fcs = []
        for i in range(len(self.args.layer_sizes) + 1):
            fc = FC(
                self.full_name(),
                size=sizes[i],
                act=acts[i],
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=w_scales[i])),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=self.init_value_)))
            self.add_sublayer('fc_%d' % i, fc)
            self.fcs.append(fc)

    def forward(self, feat_embeddings):
        y_dnn = fluid.layers.reshape(
            feat_embeddings,
            [-1, self.args.num_field * self.args.embedding_size])
        for fc in self.fcs:
            y_dnn = fc(y_dnn)
        return y_dnn
