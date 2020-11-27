#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid
import math
from paddle.fluid.param_attr import ParamAttr

__all__ = ["InceptionV3"]

class InceptionV3():
    def __init__(self):
        self.inception_a_list = [32, 64, 64]
        self.inception_c_list = [128, 160, 160, 192]

    def net(self, input, class_dim=1000):
        x = self.inception_stem(input)
        for i, pool_features in enumerate(self.inception_a_list):
            x = self.inceptionA(x, pool_features, name=str(i+1))
        x = self.inceptionB(x, name="1")
        for i, channels_7x7 in enumerate(self.inception_c_list):
             x = self.inceptionC(x, channels_7x7, name=str(i+1))
        x = self.inceptionD(x, name="1")
        x = self.inceptionE(x, name="1")
        x = self.inceptionE(x, name="2")

        pool = fluid.layers.pool2d(input=x, pool_type="avg", global_pooling=True)

        drop = fluid.layers.dropout(x=pool, dropout_prob=0.2)

        stdv = 1.0 / math.sqrt(drop.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=drop,
            size=class_dim,
            param_attr=ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv), name="fc_weights"),
            bias_attr=ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv), name="fc_offset"))
        return out

    def conv_bn_layer(self,
                      data,
                      num_filters,
                      filter_size,
                      stride=1,
                      padding=0,
                      groups=1,
                      act="relu",
                      name=None):
        conv = fluid.layers.conv2d(
            input=data,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name+"_weights"),
            bias_attr=False,
            name=name)
        return fluid.layers.batch_norm(input=conv, 
                                       act=act,
                                       param_attr = ParamAttr(name=name+"_bn_scale"),
                                       bias_attr=ParamAttr(name=name+"_bn_offset"),
                                       moving_mean_name=name+"_bn_mean",
                                       moving_variance_name=name+"_bn_variance")
    
    def inception_stem(self, x):
        x = self.conv_bn_layer(x, 
                               num_filters=32, 
                               filter_size=3, 
                               stride=2,
                               act="relu",
                               name="conv_1a_3x3")
        x = self.conv_bn_layer(x, 
                               num_filters=32, 
                               filter_size=3, 
                               stride=1,
                               act="relu",
                               name="conv_2a_3x3")        
        x = self.conv_bn_layer(x, 
                               num_filters=64, 
                               filter_size=3, 
                               padding=1,
                               act="relu",
                               name="conv_2b_3x3")         

        x = fluid.layers.pool2d(input=x, pool_size=3, pool_stride=2, pool_type="max")
        
        x = self.conv_bn_layer(x, 
                               num_filters=80, 
                               filter_size=1, 
                               act="relu",
                               name="conv_3b_1x1")  
        x = self.conv_bn_layer(x, 
                               num_filters=192, 
                               filter_size=3, 
                               act="relu",
                               name="conv_4a_3x3")  

        x = fluid.layers.pool2d(input=x, pool_size=3, pool_stride=2, pool_type="max")

        return x

    def inceptionA(self, x, pool_features, name=None):
        branch1x1 = self.conv_bn_layer(x, 
                                       num_filters=64, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_a_branch1x1_"+name)
        branch5x5 = self.conv_bn_layer(x, 
                                       num_filters=48, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_a_branch5x5_1_"+name)
        branch5x5 = self.conv_bn_layer(branch5x5, 
                                       num_filters=64, 
                                       filter_size=5, 
                                       padding=2, 
                                       act="relu",
                                       name="inception_a_branch5x5_2_"+name)

        branch3x3dbl = self.conv_bn_layer(x, 
                                       num_filters=64, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_a_branch3x3dbl_1_"+name)
        branch3x3dbl = self.conv_bn_layer(branch3x3dbl, 
                                       num_filters=96, 
                                       filter_size=3, 
                                       padding=1,
                                       act="relu",
                                       name="inception_a_branch3x3dbl_2_"+name)
        branch3x3dbl = self.conv_bn_layer(branch3x3dbl, 
                               num_filters=96, 
                               filter_size=3, 
                               padding=1,
                               act="relu",
                               name="inception_a_branch3x3dbl_3_"+name)
        branch_pool = fluid.layers.pool2d(x, pool_size=3, pool_padding=1, pool_type="avg")
        branch_pool = self.conv_bn_layer(branch_pool, 
                               num_filters=pool_features, 
                               filter_size=1, 
                               act="relu",
                               name="inception_a_branch_pool_"+name)
        
        concat = fluid.layers.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=1)
        
        return concat

    
    
    def inceptionB(self, x, name=None):
        branch3x3 = self.conv_bn_layer(x, 
                                       num_filters=384, 
                                       filter_size=3, 
                                       stride=2,
                                       act="relu",
                                       name="inception_b_branch3x3_"+name)
        branch3x3dbl = self.conv_bn_layer(x, 
                                       num_filters=64, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_b_branch3x3dbl_1_"+name)
        branch3x3dbl = self.conv_bn_layer(branch3x3dbl, 
                                       num_filters=96, 
                                       filter_size=3, 
                                       padding=1,
                                       act="relu",
                                       name="inception_b_branch3x3dbl_2_"+name)
        branch3x3dbl = self.conv_bn_layer(branch3x3dbl, 
                                       num_filters=96, 
                                       filter_size=3,
                                       stride=2,
                                       act="relu",
                                       name="inception_b_branch3x3dbl_3_"+name)
        branch_pool = fluid.layers.pool2d(x, pool_size=3, pool_stride=2, pool_type="max")
        
        
        concat = fluid.layers.concat([branch3x3, branch3x3dbl, branch_pool], axis=1)
        
        return concat
    
    
    def inceptionC(self, x, channels_7x7, name=None):
        branch1x1 = self.conv_bn_layer(x, 
                                       num_filters=192, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_c_branch1x1_"+name)
        branch7x7 = self.conv_bn_layer(x, 
                                       num_filters=channels_7x7, 
                                       filter_size=1, 
                                       stride=1,
                                       act="relu",
                                       name="inception_c_branch7x7_1_"+name)
        branch7x7 = self.conv_bn_layer(branch7x7, 
                                       num_filters=channels_7x7, 
                                       filter_size=(1, 7), 
                                       stride=1,
                                       padding=(0, 3),
                                       act="relu",
                                       name="inception_c_branch7x7_2_"+name)
        branch7x7 = self.conv_bn_layer(branch7x7, 
                                       num_filters=192, 
                                       filter_size=(7, 1), 
                                       stride=1,
                                       padding=(3, 0),
                                       act="relu",
                                       name="inception_c_branch7x7_3_"+name)
        
        branch7x7dbl = self.conv_bn_layer(x, 
                                       num_filters=channels_7x7, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_c_branch7x7dbl_1_"+name)
        branch7x7dbl = self.conv_bn_layer(branch7x7dbl, 
                                       num_filters=channels_7x7, 
                                       filter_size=(7, 1), 
                                       padding = (3, 0),
                                       act="relu",
                                       name="inception_c_branch7x7dbl_2_"+name)
        branch7x7dbl = self.conv_bn_layer(branch7x7dbl, 
                                       num_filters=channels_7x7, 
                                       filter_size=(1, 7), 
                                       padding = (0, 3),
                                       act="relu",
                                       name="inception_c_branch7x7dbl_3_"+name)
        branch7x7dbl = self.conv_bn_layer(branch7x7dbl, 
                                       num_filters=channels_7x7, 
                                       filter_size=(7, 1), 
                                       padding = (3, 0),
                                       act="relu",
                                       name="inception_c_branch7x7dbl_4_"+name)
        branch7x7dbl = self.conv_bn_layer(branch7x7dbl, 
                                       num_filters=192, 
                                       filter_size=(1, 7), 
                                       padding = (0, 3),
                                       act="relu",
                                       name="inception_c_branch7x7dbl_5_"+name)
       
        branch_pool = fluid.layers.pool2d(x, pool_size=3, pool_stride=1, pool_padding=1, pool_type="avg")
        branch_pool = self.conv_bn_layer(branch_pool, 
                                       num_filters=192, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_c_branch_pool_"+name)
        
        concat = fluid.layers.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=1)
        
        return concat

    
    def inceptionD(self, x, name=None):
        branch3x3 = self.conv_bn_layer(x, 
                                       num_filters=192, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_d_branch3x3_1_"+name)
        branch3x3 = self.conv_bn_layer(branch3x3, 
                                       num_filters=320, 
                                       filter_size=3, 
                                       stride=2,
                                       act="relu",
                                       name="inception_d_branch3x3_2_"+name)
        branch7x7x3 = self.conv_bn_layer(x, 
                                       num_filters=192, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_d_branch7x7x3_1_"+name)
        branch7x7x3 = self.conv_bn_layer(branch7x7x3, 
                                       num_filters=192, 
                                       filter_size=(1, 7), 
                                       padding=(0, 3),
                                       act="relu",
                                       name="inception_d_branch7x7x3_2_"+name)
        branch7x7x3 = self.conv_bn_layer(branch7x7x3, 
                                       num_filters=192, 
                                       filter_size=(7, 1), 
                                       padding=(3, 0),
                                       act="relu",
                                       name="inception_d_branch7x7x3_3_"+name)
        branch7x7x3 = self.conv_bn_layer(branch7x7x3, 
                                       num_filters=192, 
                                       filter_size=3, 
                                       stride=2,
                                       act="relu",
                                       name="inception_d_branch7x7x3_4_"+name)
        branch_pool = fluid.layers.pool2d(x, pool_size=3, pool_stride=2, pool_type="max")
        concat = fluid.layers.concat([branch3x3, branch7x7x3, branch_pool], axis=1)
        
        return concat
    
    
    def inceptionE(self, x, name=None):
        branch1x1 = self.conv_bn_layer(x, 
                                       num_filters=320, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_e_branch1x1_"+name)
        branch3x3 = self.conv_bn_layer(x, 
                                       num_filters=384, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_e_branch3x3_1_"+name)
        branch3x3_2a = self.conv_bn_layer(branch3x3, 
                                       num_filters=384, 
                                       filter_size=(1, 3), 
                                       padding=(0, 1),
                                       act="relu",
                                       name="inception_e_branch3x3_2a_"+name)
        branch3x3_2b = self.conv_bn_layer(branch3x3, 
                                       num_filters=384, 
                                       filter_size=(3, 1), 
                                       padding=(1, 0),
                                       act="relu",
                                       name="inception_e_branch3x3_2b_"+name)
        
        branch3x3 = fluid.layers.concat([branch3x3_2a, branch3x3_2b], axis=1)
        branch3x3dbl = self.conv_bn_layer(x, 
                                       num_filters=448, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_e_branch3x3dbl_1_"+name)
        branch3x3dbl = self.conv_bn_layer(branch3x3dbl, 
                                       num_filters=384, 
                                       filter_size=3, 
                                       padding=1,
                                       act="relu",
                                       name="inception_e_branch3x3dbl_2_"+name)
        branch3x3dbl_3a = self.conv_bn_layer(branch3x3dbl, 
                                       num_filters=384, 
                                       filter_size=(1, 3), 
                                       padding=(0, 1),
                                       act="relu",
                                       name="inception_e_branch3x3dbl_3a_"+name)
        branch3x3dbl_3b = self.conv_bn_layer(branch3x3dbl, 
                                       num_filters=384, 
                                       filter_size=(3, 1), 
                                       padding=(1, 0),
                                       act="relu",
                                       name="inception_e_branch3x3dbl_3b_"+name)
        branch3x3dbl = fluid.layers.concat([branch3x3dbl_3a, branch3x3dbl_3b], axis=1)
        branch_pool = fluid.layers.pool2d(x, pool_size=3, pool_stride=1, pool_padding=1, pool_type="avg")
        branch_pool = self.conv_bn_layer(branch_pool, 
                                       num_filters=192, 
                                       filter_size=1, 
                                       act="relu",
                                       name="inception_e_branch_pool_"+name)
        concat = fluid.layers.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=1)
        
        return concat
 
