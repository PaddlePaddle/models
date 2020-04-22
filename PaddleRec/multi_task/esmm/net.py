import numpy as np
import os
import paddle.fluid as fluid
import paddle
import utils
import args


class ESMM(object):
 
    def fc(self,tag, data, out_dim, active='prelu'):
        
        init_stddev = 1.0
        scales = 1.0  / np.sqrt(data.shape[1])
        
        p_attr = fluid.param_attr.ParamAttr(name='%s_weight' % tag,
                    initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=init_stddev * scales))
                    
        b_attr = fluid.ParamAttr(name='%s_bias' % tag, initializer=fluid.initializer.Constant(0.1))
        
        out = fluid.layers.fc(input=data,
                            size=out_dim,
                            act=active,
                            param_attr=p_attr, 
                            bias_attr =b_attr,
                            name=tag)
        return out
        
    def input_data(self):
        sparse_input_ids = [
            fluid.data(name="field_" + str(i), shape=[-1, 1], dtype="int64", lod_level=1) for i in range(0,23)
        ]
        label_ctr = fluid.data(name="ctr", shape=[-1, 1], dtype="int64")
        label_cvr = fluid.data(name="cvr", shape=[-1, 1], dtype="int64")
        inputs = sparse_input_ids + [label_ctr] + [label_cvr]
        
        return inputs
    
    def net(self,inputs,vocab_size,embed_size):
        
        emb = []
        for data in inputs[0:-2]:
            feat_emb = fluid.embedding(input=data,
                                size=[vocab_size, embed_size],
                                param_attr=fluid.ParamAttr(name='dis_emb',
                                                            learning_rate=5,
                                                            initializer=fluid.initializer.Xavier(fan_in=embed_size,fan_out=embed_size)
                                                            ),
                                is_sparse=True)
            #fluid.layers.Print(feat_emb, message="feat_emb")
            field_emb = fluid.layers.sequence_pool(input=feat_emb,pool_type='sum')
            emb.append(field_emb)
        concat_emb = fluid.layers.concat(emb, axis=1)
        
        
        # ctr
        
        active = 'relu'
        ctr_fc1 = self.fc('ctr_fc1', concat_emb, 200, active)
        ctr_fc2 = self.fc('ctr_fc2', ctr_fc1, 80, active)
        ctr_out = self.fc('ctr_out', ctr_fc2, 2, 'softmax')
        
        # cvr
        cvr_fc1 = self.fc('cvr_fc1', concat_emb, 200, active)
        cvr_fc2 = self.fc('cvr_fc2', cvr_fc1, 80, active)
        cvr_out = self.fc('cvr_out', cvr_fc2, 2,'softmax')
    
        ctr_clk = inputs[-2]
        ctcvr_buy = inputs[-1]
        #ctr_label = fluid.layers.concat(input=[ctr_clk,1-ctr_clk],axis=1)
        #ctcvr_label = fluid.layers.concat(input=[ctcvr_buy,1-ctcvr_buy],axis=1)
        #ctr_label = fluid.layers.cast(x=ctr_label, dtype='float32')
        #ctcvr_label = fluid.layers.cast(x=ctcvr_label, dtype='float32')
        
        ctr_prop_one = fluid.layers.slice(ctr_out, axes=[1], starts=[1], ends=[2])
        cvr_prop_one = fluid.layers.slice(cvr_out, axes=[1], starts=[1], ends=[2])
        
        
        ctcvr_prop_one = fluid.layers.elementwise_mul(ctr_prop_one, cvr_prop_one)
        ctcvr_prop = fluid.layers.concat(input=[1-ctcvr_prop_one,ctcvr_prop_one], axis = 1)
    
        loss_ctr = paddle.fluid.layers.cross_entropy(input=ctr_out, label=ctr_clk)
        loss_ctcvr = paddle.fluid.layers.cross_entropy(input=ctcvr_prop, label=ctcvr_buy)
        cost = loss_ctr + loss_ctcvr
        avg_cost = fluid.layers.mean(cost)
        #fluid.layers.Print(ctr_clk, message="ctr_clk")
        auc_ctr, batch_auc_ctr, auc_states_ctr = fluid.layers.auc(input=ctr_out, label=ctr_clk)
        auc_ctcvr, batch_auc_ctcvr, auc_states_ctcvr = fluid.layers.auc(input=ctcvr_prop, label=ctcvr_buy)
    
        return avg_cost,auc_ctr,auc_ctcvr
  
    




         

    
    
    
    
    
    
    
    
    
    