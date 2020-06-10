import paddle.fluid as fluid
import itertools

class Fibinet(object):
    def input_data(self, dense_feature_dim):
        dense_input = fluid.data(name="dense_input",
                                 shape=[-1, dense_feature_dim],
                                 dtype="float32")

        sparse_input_ids = [
            fluid.data(name="C" + str(i),
                       shape=[-1, 1],
                       lod_level=1,
                       dtype="int64") for i in range(1, 27)
        ]

        label = fluid.data(name="label", shape=[-1, 1], dtype="int64")

        inputs = [dense_input] + sparse_input_ids + [label]
        return inputs

    def fc(self, data, size, active, tag):
        output = fluid.layers.fc(input=data,
                            size=size, 
                            param_attr=fluid.initializer.Xavier(uniform=False),
                            act=active,
                            name=tag)
                            
        return output
    def SENETLayer(self, inputs, filed_size, reduction_ratio = 3):
        reduction_size = max(1, filed_size // reduction_ratio)
        Z = fluid.layers.reduce_mean(inputs, dim=-1)
        
        A_1 = self.fc(Z, reduction_size, 'relu', 'W_1')
        A_2 = self.fc(A_1, filed_size, 'relu', 'W_2')

        V = fluid.layers.elementwise_mul(inputs, y = fluid.layers.unsqueeze(input=A_2, axes=[2]))
        
        return fluid.layers.split(V, num_or_sections=filed_size, dim=1)

    def BilinearInteraction(self, inputs, filed_size, embedding_size, bilinear_type="interaction"):
        if bilinear_type == "all":
            p = [fluid.layers.elementwise_mul(self.fc(v_i, embedding_size, None, None), fluid.layers.squeeze(input=v_j, axes=[1])) for v_i, v_j in itertools.combinations(inputs, 2)]
        else:
            raise NotImplementedError

        return fluid.layers.concat(input=p, axis=1)

    def DNNLayer(self, inputs, dropout_rate=0.5):
        deep_input = inputs
        for i, hidden_unit in enumerate([400, 400, 400]):
            fc_out = self.fc(deep_input, hidden_unit, 'relu', 'd_' + str(i))
            fc_out = fluid.layers.dropout(fc_out, dropout_prob=dropout_rate)
            deep_input = fc_out

        return deep_input

    def net(self, inputs, sparse_feature_dim, embedding_size, reduction_ratio, bilinear_type, dropout_rate=0.5): 
        filed_size = len(inputs[1:-1])
        
        emb = []
        for data in inputs[1 :-1]:
            feat_emb = fluid.embedding(input=data,
                                size=[sparse_feature_dim, embedding_size],
                                param_attr=fluid.ParamAttr(name='dis_emb',
                                                            learning_rate=5,
                                                            initializer=fluid.initializer.Xavier(fan_in=embedding_size,fan_out=embedding_size)
                                                            ),
                                is_sparse=True)
            emb.append(feat_emb)
        concat_emb = fluid.layers.concat(emb, axis=1)

        senet_output = self.SENETLayer(concat_emb, filed_size, reduction_ratio)
        senet_bilinear_out = self.BilinearInteraction(senet_output, filed_size, embedding_size, bilinear_type)
        
        concat_emb = fluid.layers.split(concat_emb, num_or_sections=filed_size, dim=1)
        bilinear_out = self.BilinearInteraction(concat_emb, filed_size, embedding_size, bilinear_type)
        dnn_input = fluid.layers.concat(input=[senet_bilinear_out, bilinear_out, inputs[0]], axis=1)
        dnn_output = self.DNNLayer(dnn_input, dropout_rate)
        label = inputs[-1]
        y_pred = self.fc(dnn_output, 1, 'sigmoid', 'logit')
        cost = fluid.layers.log_loss(input=y_pred, label=fluid.layers.cast(x=label, dtype='float32'))
        avg_cost = fluid.layers.mean(cost)
        auc_val, batch_auc, auc_states = fluid.layers.auc(input=y_pred, label=label)
        
        return avg_cost, auc_val, batch_auc, auc_states
