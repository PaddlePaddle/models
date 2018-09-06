import datareader as reader
import math
import numpy as np
import paddle.fluid as fluid

class emlloss():
    def __init__(self, train_batch_size = 50, test_batch_size = 50, infer_batch_size = 50, samples_each_class=2, fea_dim=2048):
        self.train_reader = reader.eml_train(train_batch_size, samples_each_class)
        self.test_reader = reader.test(test_batch_size)
        self.infer_reader = reader.infer(infer_batch_size)
        self.samples_each_class = samples_each_class
        self.fea_dim = fea_dim
        self.batch_size = train_batch_size
        
    def surrogate_function(self, input):
        beta = 100000
        output = fluid.layers.log(1+beta*input)/math.log(1+beta)
        return output
 
    def surrogate_function_approximate(self, input, bias):
        beta = 100000
        output = (fluid.layers.log(beta*input)+bias)/math.log(1+beta)
        return output
  
    def generate_index(self, batch_size, samples_each_class):
        a = np.arange(0, batch_size*batch_size) # N*N x 1
        a = a.reshape(-1,batch_size) # N x N
        steps = batch_size//samples_each_class
        res = []
        for i in range(batch_size):
            step = i // samples_each_class
            start = step * samples_each_class
            end = (step + 1) * samples_each_class
            p = []
            n = []
            for j, k in enumerate(a[i]):
                if j >= start and j < end:
                    p.append(k)
                else:
                    n.append(k)
            comb = p + n
            res += comb
        res = np.array(res).astype(np.int32)
        return res

    def loss(self, input):
        samples_each_class = self.samples_each_class
        fea_dim = self.fea_dim
        batch_size = self.batch_size
        feature1 = fluid.layers.reshape(input, shape = [-1, fea_dim])
        feature2 = fluid.layers.transpose(feature1, perm = [1,0])
        ab = fluid.layers.mul(x=feature1, y=feature2)
        a2 = fluid.layers.square(feature1)
        a2 = fluid.layers.reduce_sum(a2, dim = 1)
        a2 = fluid.layers.reshape(a2, shape = [-1])
        b2 = fluid.layers.square(feature2)
        b2 = fluid.layers.reduce_sum(b2, dim = 0)
        b2 = fluid.layers.reshape(b2, shape = [-1])
 
        d = fluid.layers.elementwise_add(-2*ab, a2, axis = 0)
        d = fluid.layers.elementwise_add(d, b2, axis = 1)
        d = fluid.layers.reshape(d, shape=[-1, 1])
 
        index = self.generate_index(batch_size, samples_each_class)
        index_var = fluid.layers.create_global_var(
            shape=[batch_size*batch_size], value=0, dtype='int32', persistable=True)
        index_var = fluid.layers.assign(index, index_var)
        index_var.stop_gradient = True
 
        d = fluid.layers.gather(d, index=index_var)
        d = fluid.layers.reshape(d, shape=[-1, batch_size])
        pos, neg = fluid.layers.split(d, 
                              num_or_sections= [samples_each_class,batch_size-samples_each_class], 
                              dim=1)
        pos_max = fluid.layers.reduce_max(pos, dim=1)
        pos_max = fluid.layers.reshape(pos_max, shape=[-1, 1])
        
        pos = fluid.layers.exp(pos-pos_max)
        pos_mean = fluid.layers.reduce_mean(pos, dim=1)
        
        neg_min = fluid.layers.reduce_min(neg, dim=1)
        neg_min = fluid.layers.reshape(neg_min, shape=[-1, 1])
        neg = fluid.layers.exp(-1*(neg-neg_min))
        neg_mean = fluid.layers.reduce_mean(neg, dim=1)
        theta = fluid.layers.reshape(neg_mean * pos_mean, shape=[-1,1])
 
        max_gap = fluid.layers.fill_constant([1], dtype='float32', value=20.0)
        max_gap.stop_gradient = True
 
        target = pos_max - neg_min
        target_max = fluid.layers.elementwise_max(target, max_gap)
        target_min = fluid.layers.elementwise_min(target, max_gap)
 
        expadj_min = fluid.layers.exp(target_min)
        loss1 = self.surrogate_function(theta*expadj_min)
        loss2 = self.surrogate_function_approximate(theta, target_max)
 
        bias = fluid.layers.exp(max_gap)
        bias = self.surrogate_function(theta*bias)
 
        loss = loss1 + loss2 - bias
 
        return loss
