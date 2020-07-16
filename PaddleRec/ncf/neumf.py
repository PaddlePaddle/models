import numpy as np
import paddle.fluid as fluid
import sys
import math
from time import time

class NeuMF(object):
    def net(self, inputs, num_users, num_items, latent_dim, layers = [64,32,16,8]):
        num_layer = len(layers) #Number of layers in the MLP
        
        
        MF_Embedding_User = fluid.embedding(input=inputs[0],
                                            size=[num_users, latent_dim],
                                            param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01),
                                            is_sparse=True)
        MF_Embedding_Item = fluid.embedding(input=inputs[1],
                                        size=[num_items, latent_dim],
                                        param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01),
                                        is_sparse=True)
        
        MLP_Embedding_User = fluid.embedding(input=inputs[0],
                                            size=[num_users, int(layers[0] / 2)],
                                            param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01),
                                            is_sparse=True)
        MLP_Embedding_Item = fluid.embedding(input=inputs[1],
                                        size=[num_items, int(layers[0] / 2)],
                                        param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01),
                                        is_sparse=True)
                                        
        # MF part
        mf_user_latent = fluid.layers.flatten(x=MF_Embedding_User, axis=1)
        mf_item_latent = fluid.layers.flatten(x=MF_Embedding_Item, axis=1)
        mf_vector = fluid.layers.elementwise_mul(mf_user_latent, mf_item_latent)
        
        # MLP part 
        # The 0-th layer is the concatenation of embedding layers
        mlp_user_latent = fluid.layers.flatten(x=MLP_Embedding_User, axis=1)
        mlp_item_latent = fluid.layers.flatten(x=MLP_Embedding_Item, axis=1)
        mlp_vector = fluid.layers.concat(input=[mlp_user_latent, mlp_item_latent], axis=-1)
        
        for i in range(1, num_layer):
            mlp_vector = fluid.layers.fc(input=mlp_vector,
                                       size=layers[i],
                                       act='relu',
                                       param_attr=fluid.ParamAttr(initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=1.0 / math.sqrt(mlp_vector.shape[1])),
                                                       regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4)),
                                       name='layer_' + str(i))
                                       
        # Concatenate MF and MLP parts
        predict_vector = fluid.layers.concat(input=[mf_vector, mlp_vector], axis=-1)

        # Final prediction layer
        prediction = fluid.layers.fc(input=predict_vector,
                                    size=1,
                                    act='sigmoid',
                                    param_attr=fluid.initializer.MSRAInitializer(uniform=True), 
                                    name='prediction')
                                    
        cost = fluid.layers.log_loss(input=prediction, label=fluid.layers.cast(x=inputs[2], dtype='float32'))
        avg_cost = fluid.layers.mean(cost)
        
        return avg_cost, prediction
        
          