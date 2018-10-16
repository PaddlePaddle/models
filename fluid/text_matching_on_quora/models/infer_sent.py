import paddle.fluid as fluid
from my_layers import bi_lstm_layer
from match_layers import ElementwiseMatching

class InferSentNet():
    """
    Base on the paper: Supervised Learning of Universal Sentence Representations from Natural Language Inference Data:
    https://arxiv.org/abs/1705.02364
    """

    def __init__(self, config):
         self._config = config

    def __call__(self, seq1, seq2, label):
        return self.body(seq1, seq2, label, self._config)

    def body(self, seq1, seq2, label, config):
        """Body function"""

        seq1_rnn = self.encoder(seq1)
        seq2_rnn = self.encoder(seq2)
        seq_match = ElementwiseMatching(seq1_rnn, seq2_rnn)

        mlp_res = self.MLP(seq_match)
        prediction = fluid.layers.fc(mlp_res, size=self._config.class_dim, act='softmax')
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_cost = fluid.layers.mean(x=loss)
        acc = fluid.layers.accuracy(input=prediction, label=label)
        return avg_cost, acc, prediction

    def encoder(self, seq):
        """encoder"""

        embed = fluid.layers.embedding(
                    input=seq,
                    size=[self._config.dict_dim, self._config.emb_dim],
                    param_attr=fluid.ParamAttr(name='emb.w', trainable=self._config.word_embedding_trainable))

        bi_lstm_h = bi_lstm_layer(
                        embed,
                        rnn_hid_dim = self._config.rnn_hid_dim, 
                        name='encoder')

        bi_lstm_h = fluid.layers.dropout(bi_lstm_h, dropout_prob=self._config.droprate_lstm)
        pool = fluid.layers.sequence_pool(input=bi_lstm_h, pool_type='max')
        return pool

    def MLP(self, vec):
        if self._config.mlp_non_linear:
            drop1 = fluid.layers.dropout(vec, dropout_prob=self._config.droprate_fc)
            fc1 = fluid.layers.fc(drop1, size=512, act='tanh')
            drop2 = fluid.layers.dropout(fc1, dropout_prob=self._config.droprate_fc)
            fc2 = fluid.layers.fc(drop2, size=512, act='tanh')
            res = fluid.layers.dropout(fc2, dropout_prob=self._config.droprate_fc)
        else:
            fc1 = fluid.layers.fc(vec, size=512, act=None)
            res = fluid.layers.fc(fc1, size=512, act=None)
        return res
