import math
import paddle


class DeepFM(paddle.nn.Layer):
    def __init__(self, args):
        super(DeepFM, self).__init__()
        self.args = args
        self.init_value_ = 0.1

        self.fm = FM(args)
        self.dnn = DNN(args)

    def forward(self, raw_feat_idx, raw_feat_value, label):
        feat_idx = paddle.reshape(raw_feat_idx,
                                  [-1, 1])  # (None * num_field) * 1
        feat_value = paddle.reshape(
            raw_feat_value,
            [-1, self.args.num_field, 1])  # None * num_field * 1

        y_first_order, y_second_order, feat_embeddings = self.fm(feat_idx,
                                                                 feat_value)
        y_dnn = self.dnn(feat_embeddings)

        predict = paddle.nn.functional.sigmoid(y_first_order + y_second_order +
                                               y_dnn)

        return predict


class FM(paddle.nn.Layer):
    def __init__(self, args):
        super(FM, self).__init__()
        self.args = args
        self.init_value_ = 0.1
        self.embedding_w = paddle.nn.Embedding(
            self.args.num_feat + 1,
            1,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0, std=self.init_value_),
                regularizer=paddle.regularizer.L1Decay(self.args.reg)))
        self.embedding = paddle.nn.Embedding(
            self.args.num_feat + 1,
            self.args.embedding_size,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.args.embedding_size)))))

    def forward(self, feat_idx, feat_value):
        # -------------------- first order term  --------------------
        first_weights_re = self.embedding_w(feat_idx)
        first_weights = paddle.reshape(
            first_weights_re,
            shape=[-1, self.args.num_field, 1])  # None * num_field * 1
        y_first_order = paddle.sum(first_weights * feat_value, 1)

        # -------------------- second order term  --------------------
        feat_embeddings_re = self.embedding(feat_idx)
        feat_embeddings = paddle.reshape(
            feat_embeddings_re,
            shape=[-1, self.args.num_field, self.args.embedding_size
                   ])  # None * num_field * embedding_size
        feat_embeddings = feat_embeddings * feat_value  # None * num_field * embedding_size

        # sum_square part
        summed_features_emb = paddle.sum(feat_embeddings,
                                         1)  # None * embedding_size
        summed_features_emb_square = paddle.square(
            summed_features_emb)  # None * embedding_size

        # square_sum part
        squared_features_emb = paddle.square(
            feat_embeddings)  # None * num_field * embedding_size
        squared_sum_features_emb = paddle.sum(squared_features_emb,
                                              1)  # None * embedding_size

        y_second_order = 0.5 * paddle.sum(
            summed_features_emb_square - squared_sum_features_emb,
            1,
            keepdim=True)  # None * 1

        return y_first_order, y_second_order, feat_embeddings


class DNN(paddle.nn.Layer):
    def __init__(self, args):
        super(DNN, self).__init__()
        self.args = args
        self.init_value_ = 0.1
        sizes = [self.args.num_field * self.args.embedding_size
                 ] + self.args.layer_sizes + [1]
        acts = [self.args.act
                for _ in range(len(self.args.layer_sizes))] + [None]
        w_scales = [
            self.init_value_ / math.sqrt(float(10))
            for _ in range(len(self.args.layer_sizes))
        ] + [self.init_value_]
        self._layers = []
        for i in range(len(self.args.layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.TruncatedNormal(
                        mean=0.0, std=w_scales[i])),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.TruncatedNormal(
                        mean=0.0, std=self.init_value_)))
            #linear = getattr(paddle.nn.functional, acts[i])(linear) if acts[i] else linear
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
            self.add_sublayer('linear_%d' % i, linear)
            self._layers.append(linear)
            self._layers.append(act)

    def forward(self, feat_embeddings):
        y_dnn = paddle.reshape(
            feat_embeddings,
            [-1, self.args.num_field * self.args.embedding_size])
        for n_layer in self._layers:
            y_dnn = n_layer(y_dnn)
        return y_dnn
