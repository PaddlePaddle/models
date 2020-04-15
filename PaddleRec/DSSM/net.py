import paddle.fluid as fluid
import numpy as np
import sys
import os
import args

class DSSM(object):

    def fc(self,tag, data, out_dim, active='prelu'):
    
        xavier=fluid.initializer.Xavier(uniform=True,fan_in=data.shape[1],fan_out=out_dim)
      
        out = fluid.layers.fc(input=data,
                            size=out_dim,
                            act=active,
                            param_attr=xavier, 
                            bias_attr =xavier,
                            name=tag)
        return out
        
    def input_data(self,TRIGRAM_D):
        query = fluid.data(name="query", shape=[-1, TRIGRAM_D], dtype="float32")
        doc_pos = fluid.data(name="doc_pos", shape=[-1, TRIGRAM_D], dtype="float32")
        inputs = [query] + [doc_pos]
        return inputs
        
    #TRIGRAM_D  letter trigrim 之后的维度
    #L1_N 第一层的输出
    #L2_N 第二层的输出
    #L3_N 第三层的输出
    #Neg 负样本个数
    def net(self,inputs,TRIGRAM_D = 1000,L1_N = 300, L2_N = 300, L3_N = 128,Neg = 4,batch_size = 1):
    
        active = 'tanh'
        #第一层
        query_l1 = self.fc('query_l1', inputs[0], L1_N,active)
        doc_pos_l1 = self.fc('doc_pos_l1', inputs[1], L1_N,active)
    
        #第二层
        query_l2 = self.fc('query_l2', query_l1, L2_N,active)
        doc_pos_l2 = self.fc('doc_l2', doc_pos_l1, L2_N,active)
        
        #第二层
        query_l3 = self.fc('query_l3', query_l2, L3_N,active)
        doc_pos_l3 = self.fc('doc_l3', doc_pos_l2, L3_N,active)
        
        temp = doc_pos_l3
        for i in range(Neg):
            rand = 0
            while(rand == 0):
                rand = int((np.random.random() + i) * batch_size / Neg)
                
            doc_pos_l3 = fluid.layers.concat(input=[doc_pos_l3,fluid.layers.slice(temp, axes=[0,1], starts=[rand, 0], ends=[batch_size,L3_N]),
                fluid.layers.slice(temp, axes=[0,1], starts=[0, 0],ends=[rand,L3_N])], axis=0)
        
            
        #计算query和doc_pos的相似度
        query_norm = fluid.layers.expand(fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(query_l3), dim=1, keep_dim=True)),expand_times=[Neg + 1,1])
        doc_norm = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(doc_pos_l3), dim=1, keep_dim=True))
        
        #prod = tf.reduce_sum(tf.mul(tf.tile(query_y, [NEG + 1, 1]), doc_y), 1, True)
        prod = fluid.layers.reduce_sum(fluid.layers.elementwise_mul(fluid.layers.expand(query_l3,expand_times=[Neg + 1,1]),doc_pos_l3), dim=1, keep_dim=True)
        norm_prod = fluid.layers.elementwise_mul(query_norm, doc_norm)
        cos_sim_raw = fluid.layers.elementwise_div(prod,norm_prod)
        
        cos_sim = fluid.layers.transpose(fluid.layers.reshape(fluid.layers.transpose(cos_sim_raw, perm=[1, 0]),shape=[Neg + 1, batch_size]),perm=[1, 0])
        
        prob = fluid.layers.softmax(cos_sim,axis=1)
        hit_prob = fluid.layers.slice(prob, axes=[0,1], starts=[0,0], ends=[batch_size,1])
        loss  = -fluid.layers.reduce_sum(fluid.layers.log(hit_prob))
        avg_loss = fluid.layers.mean(x=loss)
        
        return avg_loss














