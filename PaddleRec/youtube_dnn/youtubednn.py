import paddle
import io
import math
import numpy as np
import paddle.fluid as fluid

class YoutubeDNN(object):
    def input_data(self, watch_vec_size, search_vec_size, other_feat_size):
        watch_vec = fluid.data(name="watch_vec", shape=[None, watch_vec_size], dtype="float32")
        search_vec = fluid.data(name="search_vec", shape=[None, search_vec_size], dtype="float32")
        other_feat = fluid.data(name="other_feat", shape=[None, other_feat_size], dtype="float32")
        label = fluid.data(name="label", shape=[None, 1], dtype="int64")

        inputs = [watch_vec] + [search_vec] + [other_feat] + [label]

        return inputs
        
    def fc(self, tag, data, out_dim, active='relu'):
        init_stddev = 1.0
        scales = 1.0  / np.sqrt(data.shape[1])
        
        if tag == 'l4':
            p_attr = fluid.param_attr.ParamAttr(name='%s_weight' % tag,
                        initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=init_stddev * scales))
        else:
            p_attr = None
                    
        b_attr = fluid.ParamAttr(name='%s_bias' % tag, initializer=fluid.initializer.Constant(0.1))

        out = fluid.layers.fc(input=data,
                            size=out_dim,
                            act=active,
                            param_attr=p_attr, 
                            bias_attr =b_attr,
                            name=tag)
        return out

    def net(self, inputs, output_size, layers=[128, 64, 32]):
        concat_feats = fluid.layers.concat(input=inputs[:-1], axis=-1)

        l1 = self.fc('l1', concat_feats, layers[0], 'relu')
        l2 = self.fc('l2', l1, layers[1], 'relu')
        l3 = self.fc('l3', l2, layers[2], 'relu')
        l4 = self.fc('l4', l3, output_size, 'softmax')

        num_seqs = fluid.layers.create_tensor(dtype='int64')
        acc = fluid.layers.accuracy(input=l4, label=inputs[-1], total=num_seqs)

        cost = fluid.layers.cross_entropy(input=l4, label=inputs[-1])
        avg_cost = fluid.layers.mean(cost)

        return avg_cost, acc, l3
