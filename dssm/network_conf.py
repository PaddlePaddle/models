from paddle import v2 as paddle
from paddle.v2.attr import ParamAttr
from utils import TaskType, logger


class DSSM(object):
    def __init__(self,
                 dnn_dims=[],
                 vocab_sizes=[],
                 task_type=TaskType.CLASSFICATION,
                 share_semantic_generator=False,
                 class_num=None,
                 share_embed=False):
        '''
        @dnn_dims: list of int
            dimentions of each layer in semantic vector generator.
        @vocab_sizes: 2-d tuple
            size of both left and right items.
        @task_type: str
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
        self.task_type = task_type
        self.class_num = class_num

        logger.info("vocabulary sizes: %s" % str(self.vocab_sizes))

    def __call__(self):
        if self.task_type == TaskType.CLASSFICATION:
            return self.build_classification_model()
        return self.build_rank_model()

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

    def build_classification_model(self):
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

        outputs = []
        for id, input in enumerate(word_vecs):
            x = self.create_fc(input, prefix=prefixs[id])
            outputs.append(x)

        concated_vector = paddle.layer.concat(outputs)
        prediction = paddle.layer.fc(
            input=concated_vector,
            size=self.class_num,
            act=paddle.activation.Softmax())
        cost = paddle.layer.classification_cost(input=prediction, label=label)
        return cost, prediction, label

    def build_rank_model(self):
        '''
        Build a pairwise rank model, and the cost is returned.

        A pairwise rank model has 3 inputs:
          - source sentence
          - left_target sentence
          - right_target sentence
          - label, 1 if left_target should be sorted in front of right_target, otherwise 0.
        '''
        pass
