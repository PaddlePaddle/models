from paddle import v2 as paddle
from paddle.v2.attr import ParamAttr
from utils import TaskType, logger, ModelType, ModelArch


class DSSM(object):
    def __init__(self,
                 dnn_dims=[],
                 vocab_sizes=[],
                 model_type=ModelType.create_classification(),
                 model_arch=ModelArch.create_cnn(),
                 share_semantic_generator=False,
                 class_num=None,
                 share_embed=False,
                 is_infer=False):
        """
        :param dnn_dims: The dimention of each layer in the semantic vector
                         generator.
        :type dnn_dims: list of int
        :param vocab_sizes: The size of left and right items.
        :type vocab_sizes: A list having 2 elements.
        :param model_type: The type of task to train the DSSM model. The value
                           should be "rank: 0", "regression: 1" or
                           "classification: 2".
        :type model_type: int
        :param model_arch: A value indicating the model architecture to use.
        :type model_arch: int
        :param share_semantic_generator: A flag indicating whether to share the
                                         semantic vector between the left and
                                         the right item.
        :type share_semantic_generator: bool
        :param share_embed: A floag indicating whether to share the embeddings
                            between the left and the right item.
        :type share_embed: bool
        :param class_num: The number of categories.
        :type class_num: int
        """
        assert len(vocab_sizes) == 2, (
            "The vocab_sizes specifying the sizes left and right inputs. "
            "Its dimension should be 2.")
        assert len(dnn_dims) > 1, ("In the DNN model, more than two layers "
                                   "are needed.")

        self.dnn_dims = dnn_dims
        self.vocab_sizes = vocab_sizes
        self.share_semantic_generator = share_semantic_generator
        self.share_embed = share_embed
        self.model_type = ModelType(model_type)
        self.model_arch = ModelArch(model_arch)
        self.class_num = class_num
        self.is_infer = is_infer
        logger.warning("Build DSSM model with config of %s, %s" %
                       (self.model_type, self.model_arch))
        logger.info("The vocabulary size is : %s" % str(self.vocab_sizes))

        # bind model architecture
        _model_arch = {
            "cnn": self.create_cnn,
            "fc": self.create_fc,
            "rnn": self.create_rnn,
        }

        def _model_arch_creater(emb, prefix=""):
            sent_vec = _model_arch.get(str(model_arch))(emb, prefix)
            dnn = self.create_dnn(sent_vec, prefix)
            return dnn

        self.model_arch_creater = _model_arch_creater

        _model_type = {
            "classification": self._build_classification_model,
            "rank": self._build_rank_model,
            "regression": self._build_regression_model,
        }
        print("model type: ", str(self.model_type))
        self.model_type_creater = _model_type[str(self.model_type)]

    def __call__(self):
        return self.model_type_creater()

    def create_embedding(self, input, prefix=""):
        """
        Create word embedding. The `prefix` is added in front of the name of
        embedding"s learnable parameter.
        """
        logger.info("Create embedding table [%s] whose dimention is %d. " %
                    (prefix, self.dnn_dims[0]))
        emb = paddle.layer.embedding(
            input=input,
            size=self.dnn_dims[0],
            param_attr=ParamAttr(name="%s_emb.w" % prefix))
        return emb

    def create_fc(self, emb, prefix=""):
        """
        A multi-layer fully connected neural networks.

        :param emb: The output of the embedding layer
        :type emb: paddle.layer
        :param prefix: A prefix will be added to the layers' names.
        :type prefix: str
        """
        _input_layer = paddle.layer.pooling(
            input=emb, pooling_type=paddle.pooling.Max())
        fc = paddle.layer.fc(input=_input_layer,
                             size=self.dnn_dims[1],
                             param_attr=ParamAttr(name="%s_fc.w" % prefix),
                             bias_attr=ParamAttr(
                                 name="%s_fc.b" % prefix, initial_std=0.))
        return fc

    def create_rnn(self, emb, prefix=""):
        """
        A GRU sentence vector learner.
        """
        gru = paddle.networks.simple_gru(
            input=emb,
            size=self.dnn_dims[1],
            mixed_param_attr=ParamAttr(name="%s_gru_mixed.w" % prefix),
            mixed_bias_param_attr=ParamAttr(name="%s_gru_mixed.b" % prefix),
            gru_param_attr=ParamAttr(name="%s_gru.w" % prefix),
            gru_bias_attr=ParamAttr(name="%s_gru.b" % prefix))
        sent_vec = paddle.layer.last_seq(gru)
        return sent_vec

    def create_cnn(self, emb, prefix=""):
        """
        A multi-layer CNN.

        :param emb: The word embedding.
        :type emb: paddle.layer
        :param prefix: The prefix will be added to of layers' names.
        :type prefix: str
        """

        def create_conv(context_len, hidden_size, prefix):
            key = "%s_%d_%d" % (prefix, context_len, hidden_size)
            conv = paddle.networks.sequence_conv_pool(
                input=emb,
                context_len=context_len,
                hidden_size=hidden_size,
                # set parameter attr for parameter sharing
                context_proj_param_attr=ParamAttr(name=key + "contex_proj.w"),
                fc_param_attr=ParamAttr(name=key + "_fc.w"),
                fc_bias_attr=ParamAttr(name=key + "_fc.b"),
                pool_bias_attr=ParamAttr(name=key + "_pool.b"))
            return conv

        logger.info("create a sequence_conv_pool whose context width is 3.")
        conv_3 = create_conv(3, self.dnn_dims[1], "cnn")
        logger.info("create a sequence_conv_pool whose context width is 4.")
        conv_4 = create_conv(4, self.dnn_dims[1], "cnn")

        return paddle.layer.concat(input=[conv_3, conv_4])

    def create_dnn(self, sent_vec, prefix):
        # if more than three layers, than a fc layer will be added.
        if len(self.dnn_dims) > 1:
            _input_layer = sent_vec
            for id, dim in enumerate(self.dnn_dims[1:]):
                name = "%s_fc_%d_%d" % (prefix, id, dim)
                logger.info("create fc layer [%s] which dimention is %d" %
                            (name, dim))
                fc = paddle.layer.fc(input=_input_layer,
                                     size=dim,
                                     act=paddle.activation.Tanh(),
                                     param_attr=ParamAttr(name="%s.w" % name),
                                     bias_attr=ParamAttr(
                                         name="%s.b" % name, initial_std=0.))
                _input_layer = fc
        return _input_layer

    def _build_classification_model(self):
        logger.info("build classification model")
        assert self.model_type.is_classification()
        return self._build_classification_or_regression_model(
            is_classification=True)

    def _build_regression_model(self):
        logger.info("build regression model")
        assert self.model_type.is_regression()
        return self._build_classification_or_regression_model(
            is_classification=False)

    def _build_rank_model(self):
        """
        Build a pairwise rank model, and the cost is returned.

        A pairwise rank model has 3 inputs:
          - source sentence
          - left_target sentence
          - right_target sentence
          - label, 1 if left_target should be sorted in front of
                   right_target, otherwise 0.
        """
        logger.info("build rank model")
        assert self.model_type.is_rank()
        source = paddle.layer.data(
            name="source_input",
            type=paddle.data_type.integer_value_sequence(self.vocab_sizes[0]))
        left_target = paddle.layer.data(
            name="left_target_input",
            type=paddle.data_type.integer_value_sequence(self.vocab_sizes[1]))
        right_target = paddle.layer.data(
            name="right_target_input",
            type=paddle.data_type.integer_value_sequence(self.vocab_sizes[1]))
        if not self.is_infer:
            label = paddle.layer.data(
                name="label_input", type=paddle.data_type.integer_value(1))

        prefixs = "_ _ _".split(
        ) if self.share_semantic_generator else "source target target".split()
        embed_prefixs = "_ _ _".split(
        ) if self.share_embed else "source target target".split()

        word_vecs = []
        for id, input in enumerate([source, left_target, right_target]):
            x = self.create_embedding(input, prefix=embed_prefixs[id])
            word_vecs.append(x)

        semantics = []
        for id, input in enumerate(word_vecs):
            x = self.model_arch_creater(input, prefix=prefixs[id])
            semantics.append(x)

        # The cosine similarity score of source and left_target.
        left_score = paddle.layer.cos_sim(semantics[0], semantics[1])
        # The cosine similarity score of source and right target.
        right_score = paddle.layer.cos_sim(semantics[0], semantics[2])

        if not self.is_infer:
            # rank cost
            cost = paddle.layer.rank_cost(left_score, right_score, label=label)
            # prediction = left_score - right_score
            # but this operator is not supported currently.
            # so AUC will not used.
            return cost, None, label
        return right_score

    def _build_classification_or_regression_model(self, is_classification):
        """
        Build a classification/regression model, and the cost is returned.

        The classification/regression task expects 3 inputs:
          - source sentence
          - target sentence
          - classification label

        """
        if is_classification:
            assert self.class_num

        source = paddle.layer.data(
            name="source_input",
            type=paddle.data_type.integer_value_sequence(self.vocab_sizes[0]))
        target = paddle.layer.data(
            name="target_input",
            type=paddle.data_type.integer_value_sequence(self.vocab_sizes[1]))
        label = paddle.layer.data(
            name="label_input",
            type=paddle.data_type.integer_value(self.class_num)
            if is_classification else paddle.data_type.dense_vector(1))

        prefixs = "_ _".split(
        ) if self.share_semantic_generator else "source target".split()
        embed_prefixs = "_ _".split(
        ) if self.share_embed else "source target".split()

        word_vecs = []
        for id, input in enumerate([source, target]):
            x = self.create_embedding(input, prefix=embed_prefixs[id])
            word_vecs.append(x)

        semantics = []
        for id, input in enumerate(word_vecs):
            x = self.model_arch_creater(input, prefix=prefixs[id])
            semantics.append(x)

        if is_classification:
            concated_vector = paddle.layer.concat(semantics)
            prediction = paddle.layer.fc(input=concated_vector,
                                         size=self.class_num,
                                         act=paddle.activation.Softmax())
            cost = paddle.layer.classification_cost(
                input=prediction, label=label)
        else:
            prediction = paddle.layer.cos_sim(*semantics)
            cost = paddle.layer.square_error_cost(prediction, label)

        if not self.is_infer:
            return cost, prediction, label
        return prediction
