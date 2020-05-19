from paddle import fluid
import utils
import numpy as np

class BiRNN(object):
    def input_data(self, item_len):
        user_slot_names = fluid.data(name='user_slot_names', shape=[None, 1], dtype='int64', lod_level=1)
        item_slot_names = fluid.data(name='item_slot_names', shape=[None, item_len], dtype='int64', lod_level=1)
        lens = fluid.data(name='lens', shape=[None], dtype='int64')
        labels = fluid.data(name='labels', shape=[None, item_len], dtype='int64', lod_level=1)

        inputs = [user_slot_names] + [item_slot_names] + [lens] + [labels]
        
        return inputs

    def default_normal_initializer(self, nf=128):
        return fluid.initializer.TruncatedNormal(loc=0.0, scale=np.sqrt(1.0/nf))

    def default_param_clip(self):
        return fluid.clip.GradientClipByValue(1.0)

    def default_regularizer(self):
        return None

    def default_fc(self, data, size, num_flatten_dims=1, act=None, name=None):
        return fluid.layers.fc(input=data,
                            size=size,
                            num_flatten_dims=num_flatten_dims,
                            param_attr=fluid.ParamAttr(initializer=self.default_normal_initializer(size),
                                                    gradient_clip=self.default_param_clip(),
                                                    regularizer=self.default_regularizer()),
                            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                    gradient_clip=self.default_param_clip(),
                                                    regularizer=self.default_regularizer()),
                            act=act,
                            name=name)

    def default_embedding(self, data, vocab_size, embed_size):
        gradient_clip = self.default_param_clip()
        reg = fluid.regularizer.L2Decay(1e-5)   # IMPORTANT, to prevent overfitting.
        embed = fluid.embedding(input=data,
                                size=[vocab_size, embed_size],
                                param_attr=fluid.ParamAttr(initializer=fluid.initializer.Xavier(),
                                                    gradient_clip=gradient_clip,
                                                    regularizer=reg),
                                is_sparse=True)

        return embed

    def default_drnn(self, data, nf, is_reverse, h_0):
        return fluid.layers.dynamic_gru(input=data,
                                        size=nf,
                                        param_attr=fluid.ParamAttr(initializer=self.default_normal_initializer(nf),
                                                            gradient_clip=self.default_param_clip(),
                                                            regularizer=self.default_regularizer()),
                                        bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                            gradient_clip=self.default_param_clip(),
                                                            regularizer=self.default_regularizer()),
                                        is_reverse=is_reverse,
                                        h_0=h_0)

    def net(self, inputs, hidden_size, user_vocab, item_vocab, embed_size):
        #encode
        user_embedding = self.default_embedding(inputs[0], user_vocab, embed_size)
        user_feature = self.default_fc(data=user_embedding,
                                        size=hidden_size,
                                        num_flatten_dims=1,
                                        act='relu', 
                                        name='user_feature_fc')

        item_embedding = self.default_embedding(inputs[1], item_vocab, embed_size)
        item_embedding = fluid.layers.sequence_unpad(x=item_embedding, length=inputs[2])
       
        item_fc = self.default_fc(data=item_embedding, 
                                    size=hidden_size, 
                                    num_flatten_dims=1, 
                                    act='relu', 
                                    name='item_fc')
        
        pos = utils.fluid_sequence_get_pos(item_fc)
        pos_embed = self.default_embedding(pos, user_vocab, embed_size)
        pos_embed = fluid.layers.squeeze(pos_embed, [1])
  
        # item gru
        gru_input = self.default_fc(data=fluid.layers.concat([item_fc, pos_embed], 1),
                                    size=hidden_size * 3,
                                    num_flatten_dims=1,
                                    act='relu',
                                    name='item_gru_fc')

        item_gru_forward = self.default_drnn(data=gru_input,
                                            nf=hidden_size,
                                            h_0=user_feature,
                                            is_reverse=False)

        item_gru_backward = self.default_drnn(data=gru_input,
                                            nf=hidden_size,
                                            h_0=user_feature,
                                            is_reverse=True)
        item_gru = fluid.layers.concat([item_gru_forward, item_gru_backward], axis=1)

        out_click_fc1 = self.default_fc(data=item_gru,
                                        size=hidden_size,
                                        num_flatten_dims=1,
                                        act='relu',
                                        name='out_click_fc1')

        click_prob = self.default_fc(data=out_click_fc1,
                                    size=2,
                                    num_flatten_dims=1,
                                    act='softmax',
                                    name='out_click_fc2')

        labels = fluid.layers.sequence_unpad(x=inputs[3], length=inputs[2])
        loss = fluid.layers.reduce_mean(fluid.layers.cross_entropy(input=click_prob, label=labels))
        auc_val, batch_auc, auc_states = fluid.layers.auc(input=click_prob, label=labels)

        return loss, auc_val, batch_auc, auc_states
