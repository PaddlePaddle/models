import numpy as np
import paddle.fluid as fluid
import sys
import math
from time import time

class GMF(object):
    def net(self, inputs, num_users, num_items, latent_dim):
        MF_Embedding_User = fluid.embedding(input=inputs[0],
                                            size=[num_users, latent_dim],
                                            param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01),
                                            is_sparse=True)
        MF_Embedding_Item = fluid.embedding(input=inputs[1],
                                        size=[num_items, latent_dim],
                                        param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01),
                                        is_sparse=True)
        
        predict_vector = fluid.layers.elementwise_mul(MF_Embedding_User, MF_Embedding_Item)
        
        prediction = fluid.layers.fc(input=predict_vector,
                                    size=1,
                                    act='sigmoid',
                                    param_attr=fluid.initializer.MSRAInitializer(uniform=True), 
                                    name='prediction')
                                    
        cost = fluid.layers.log_loss(input=prediction, label=fluid.layers.cast(x=inputs[2], dtype='float32'))
        avg_cost = fluid.layers.mean(cost)
        
        return avg_cost, prediction

        