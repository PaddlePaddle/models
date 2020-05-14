import numpy as np
import paddle.fluid as fluid
import sys
import math
from time import time

class MLP(object):
    def net(self, inputs, num_users, num_items, layers = [20, 10]):
        
        num_layer = len(layers) #Number of layers in the MLP
        
        MLP_Embedding_User = fluid.embedding(input=inputs[0],
                                            size=[num_users, int(layers[0] / 2)],
                                            param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01),
                                            is_sparse=True)
        MLP_Embedding_Item = fluid.embedding(input=inputs[1],
                                        size=[num_items, int(layers[0] / 2)],
                                        param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01),
                                        is_sparse=True)
        
        # The 0-th layer is the concatenation of embedding layers
        vector = fluid.layers.concat(input=[MLP_Embedding_User, MLP_Embedding_Item], axis=-1)
        
        for i in range(1, num_layer):
            vector = fluid.layers.fc(input=vector,
                                       size=layers[i],
                                       act='relu',
                                       param_attr=fluid.ParamAttr(initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=1.0 / math.sqrt(vector.shape[1])),
                                                       regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4)),
                                       name='layer_' + str(i))
                                       
        # Final prediction layer
        prediction = fluid.layers.fc(input=vector,
                                    size=1,
                                    act='sigmoid',
                                    param_attr=fluid.initializer.MSRAInitializer(uniform=True), 
                                    name='prediction')
                                    
        cost = fluid.layers.log_loss(input=prediction, label=fluid.layers.cast(x=inputs[2], dtype='float32'))
        avg_cost = fluid.layers.mean(cost)
        
        return avg_cost, prediction

                