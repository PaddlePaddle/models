
import paddle.fluid as fluid

class cdssmNet():
    """cdssm net"""

    def __init__(self, config):
         self._config = config

    def __call__(self, seq1, seq2, label):
        return self.body(seq1, seq2, label, self._config)

    def body(self, seq1, seq2, label, config):
        """Body function"""
        
        def conv_model(seq):
            embed = fluid.layers.embedding(input=seq, size=[config.dict_dim, config.emb_dim], param_attr='emb.w')
            conv = fluid.layers.sequence_conv(embed,
                                        num_filters=config.kernel_count,
                                        filter_size=config.kernel_size,
                                        filter_stride=1,
                                        padding=True, # TODO: what is padding
                                        bias_attr=False,
                                        param_attr='conv1d.w',
                                        act='relu')
            #print paddle.parameters.get('conv1d.w').shape

            conv = fluid.layers.dropout(conv, dropout_prob = config.droprate_conv)
            pool = fluid.layers.sequence_pool(conv, pool_type="max")
            fc = fluid.layers.fc(pool,
                             size=config.fc_dim,
                             param_attr='fc1.w',
                             bias_attr='fc1.b',
                             act='relu')
            return fc
 
        def MLP(vec):
            for dim in config.mlp_hid_dim:
                vec = fluid.layers.fc(vec, size=dim, act='relu')
                vec = fluid.layers.dropout(vec, dropout_prob=config.droprate_fc)
            return vec

        seq1_fc = conv_model(seq1)
        seq2_fc = conv_model(seq2)
        concated_seq = fluid.layers.concat(input=[seq1_fc, seq2_fc], axis=1)
        mlp_res = MLP(concated_seq)
        prediction = fluid.layers.fc(mlp_res, size=config.class_dim, act='softmax')
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_cost = fluid.layers.mean(x=loss)
        acc = fluid.layers.accuracy(input=prediction, label=label)
        return avg_cost, acc, prediction

