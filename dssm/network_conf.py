from paddle import v2 as paddle
from paddle.v2.attr import ParamAttr
from utils import TaskType, logger, ModelType


class DSSM(object):
    def __init__(self,
                 dnn_dims=[],
                 vocab_sizes=[],
                 model_type=ModelType.CLASSIFICATION,
                 share_semantic_generator=False,
                 class_num=None,
                 share_embed=False):
        '''
        @dnn_dims: list of int
            dimentions of each layer in semantic vector generator.
        @vocab_sizes: 2-d tuple
            size of both left and right items.
        @model_type: str
            type of task, should be 'rank', 'regression' or 'classification'
        @share_semantic_generator: bool
            whether to share the semantic vector generator for both left and right.
        @share_embed: bool
            whether to share the embeddings between left and right.
        @class_num: int
            number of categories.
        '''
        assert len(
            vocab_sizes
        ) == 2, "vocab_sizes specify the sizes left and right inputs, and dim should be 2."

        self.dnn_dims = dnn_dims
        self.vocab_sizes = vocab_sizes
        self.share_semantic_generator = share_semantic_generator
        self.share_embed = share_embed
        self.model_type = model_type
        self.class_num = class_num

        logger.info("vocabulary sizes: %s" % str(self.vocab_sizes))

    def __call__(self):
        if self.model_type == ModelType.CLASSIFICATION:
            return self._build_classification_model()
        return self._build_rank_model()

    def create_embedding(self, input, prefix=''):
        '''
        Create an embedding table whose name has a `prefix`.
        '''
        emb = paddle.layer.embedding(
            input=input,
            size=self.dnn_dims[0],
            param_attr=ParamAttr(name='%s_emb.w' % prefix))
        return emb

    def create_fc(self, emb, prefix=''):
        '''
        A multi-layer fully connected neural networks.

        @emb: paddle.layer
            output of the embedding layer
        @prefix: str
            prefix of layers' names, used to share parameters between more than one `fc` parts.
        '''
        _input_layer = paddle.layer.pooling(
            input=emb, pooling_type=paddle.pooling.Max())
        for id, dim in enumerate(self.dnn_dims[1:]):
            name = "%s_fc_%d_%d" % (prefix, id, dim)
            fc = paddle.layer.fc(
                name=name,
                input=_input_layer,
                size=dim,
                act=paddle.activation.Relu(),
                param_attr=ParamAttr(name='%s.w' % name),
                bias_attr=None, )
            _input_layer = fc
        return _input_layer

    def create_cnn(self, emb, prefix=''):
        '''
        A multi-layer CNN.

        @emb: paddle.layer
            output of the embedding layer
        @prefix: str
            prefix of layers' names, used to share parameters between more than one `cnn` parts.
        '''
        pass

    def _build_classification_model(self):
        '''
        Build a classification model, and the cost is returned.

        A Classification has 3 inputs:
          - source sentence
          - target sentence
          - classification label

        '''
        # prepare inputs.
        assert self.class_num

        source = paddle.layer.data(
            name='source_input',
            type=paddle.data_type.integer_value_sequence(self.vocab_sizes[0]))
        target = paddle.layer.data(
            name='target_input',
            type=paddle.data_type.integer_value_sequence(self.vocab_sizes[1]))
        label = paddle.layer.data(
            name='label_input',
            type=paddle.data_type.integer_value(self.class_num))

        prefixs = '_ _'.split(
        ) if self.share_semantic_generator else 'left right'.split()
        embed_prefixs = '_ _'.split(
        ) if self.share_embed else 'left right'.split()

        word_vecs = []
        for id, input in enumerate([source, target]):
            x = self.create_embedding(input, prefix=embed_prefixs[id])
            word_vecs.append(x)

        semantics = []
        for id, input in enumerate(word_vecs):
            x = self.create_fc(input, prefix=prefixs[id])
            semantics.append(x)

        concated_vector = paddle.layer.concat(semantics)
        prediction = paddle.layer.fc(
            input=concated_vector,
            size=self.class_num,
            act=paddle.activation.Softmax())
        cost = paddle.layer.classification_cost(input=prediction, label=label)
        return cost, prediction, label

    def _build_rank_model(self):
        '''
        Build a pairwise rank model, and the cost is returned.

        A pairwise rank model has 3 inputs:
          - source sentence
          - left_target sentence
          - right_target sentence
          - label, 1 if left_target should be sorted in front of right_target, otherwise 0.
        '''
        source = paddle.layer.data(
            name='source_input',
            type=paddle.data_type.integer_value_sequence(self.vocab_sizes[0]))
        left_target = paddle.layer.data(
            name='left_target_input',
            type=paddle.data_type.integer_value_sequence(self.vocab_sizes[1]))
        right_target = paddle.layer.data(
            name='right_target_input',
            type=paddle.data_type.integer_value_sequence(self.vocab_sizes[1]))
        label = paddle.layer.data(
            name='label_input', type=paddle.data_type.integer_value(1))

        prefixs = '_ _ _'.split(
        ) if self.share_semantic_generator else 'source left right'.split()
        embed_prefixs = '_ _'.split(
        ) if self.share_embed else 'source target target'.split()

        word_vecs = []
        for id, input in enumerate([source, left_target, right_target]):
            x = self.create_embedding(input, prefix=embed_prefixs[id])
            word_vecs.append(x)

        semantics = []
        for id, input in enumerate(word_vecs):
            x = self.create_fc(input, prefix=prefixs[id])
            semantics.append(x)

        # cossim score of source and left_target
        left_score = paddle.layer.cos_sim(semantics[0], semantics[1])
        # cossim score of source and right target
        right_score = paddle.layer.cos_sim(semantics[0], semantics[2])

        # rank cost
        cost = paddle.layer.rank_cost(left_score, right_score, label=label)
        # prediction = left_score - right_score
        # but this operator is not supported currently.
        # so AUC will not used.
        return cost, None, None


class RankMetrics(object):
    '''
    A custom metrics to calculate AUC.

    Paddle's rank model do not support auc evaluator directly,
    to make it, infer all the outputs and use python to calculate
    the metrics.
    '''

    def __init__(self, model_parameters, left_score_layer, right_score_layer,
                 label):
        '''
        @model_parameters: dict
            model's parameters
        @left_score_layer: paddle.layer
            left part's score
        @right_score_laeyr: paddle.layer
            right part's score
        @label: paddle.data_layer
            label input
        '''
        self.inferer = paddle.inference.Inference(
            output_layer=[left_score_layer, right_score_layer],
            parameters=model_parameters)

    def test(self, input):
        scores = []
        for id, rcd in enumerate(input()):
            # output [left_score, right_score, label]
            res = self.inferer(input=input)
            scores.append(res)
        print scores
