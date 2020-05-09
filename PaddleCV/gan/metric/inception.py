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

import math
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

__all__ = ['InceptionV3']


class InceptionV3:
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self, output_blocks = [DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True):
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, 'Last possible output block index is 3'

    def network(self, x, class_dim=1000, aux_logits=False):
        if self.resize_input:
            x = fluid.layers.resize_bilinear(x, out_shape=[299, 299], align_corners=False, align_mode=0)

        if self.normalize_input:
            x = x * 2 - 1

        out, _, = self.fid_inceptionv3(x, class_dim, aux_logits)
        return out
        

    def fid_inceptionv3(self, x, num_classes=1000, aux_logits=False):
        """ inception v3 model for FID computation
        """
        out = []
        aux = None

        ### block0
        x = self.conv_bn_layer(x, 32, 3, stride=2, name='Conv2d_1a_3x3')
        x = self.conv_bn_layer(x, 32, 3, name='Conv2d_2a_3x3')
        x = self.conv_bn_layer(x, 64, 3, padding=1, name='Conv2d_2b_3x3')
        x = fluid.layers.pool2d(x, pool_size=3, pool_stride=2, pool_type='max')
        if 0 in self.output_blocks:
            out.append(x)

        if self.last_needed_block >= 1:
            ### block1
            x = self.conv_bn_layer(x, 80, 1, name='Conv2d_3b_1x1')
            x = self.conv_bn_layer(x, 192, 3, name='Conv2d_4a_3x3')
            x = fluid.layers.pool2d(x, pool_size=3, pool_stride=2, pool_type='max')
            if 1 in self.output_blocks:
                out.append(x)

        if self.last_needed_block >= 2:
            ### block2
            ### Mixed_5b 5c 5d
            x = self.fid_inceptionA(x, pool_features=32, name='Mixed_5b')
            x = self.fid_inceptionA(x, pool_features=64, name='Mixed_5c')
            x = self.fid_inceptionA(x, pool_features=64, name='Mixed_5d')

            ### Mixed_6
            x = self.inceptionB(x, name='Mixed_6a')
            x = self.fid_inceptionC(x, c7=128, name='Mixed_6b')
            x = self.fid_inceptionC(x, c7=160, name='Mixed_6c')
            x = self.fid_inceptionC(x, c7=160, name='Mixed_6d')
            x = self.fid_inceptionC(x, c7=192, name='Mixed_6e')
            if 2 in self.output_blocks:
                out.append(x)

            if aux_logits:
                aux = self.inceptionAux(x, num_classes, name='AuxLogits')

        if self.last_needed_block >= 3:
            ### block3
            ### Mixed_7
            x = self.inceptionD(x, name='Mixed_7a')
            x = self.fid_inceptionE_1(x, name='Mixed_7b')
            x = self.fid_inceptionE_2(x, name='Mixed_7c')

            x = fluid.layers.pool2d(x, global_pooling=True, pool_type='avg')
            out.append(x)

            #x = fluid.layers.dropout(x, dropout_prob=0.5)
            #x = fluid.layers.flatten(x, axis=1)
            #x = fluid.layers.fc(x, size=num_classes, param_attr=ParamAttr(name='fc.weight'), bias_attr=ParamAttr(name='fc.bias'))

        return out, aux

    def inceptionA(self, x, pool_features, name=None):
        branch1x1 = self.conv_bn_layer(x, 64, 1, name=name+'.branch1x1')

        branch5x5 = self.conv_bn_layer(x, 48, 1, name=name+'.branch5x5_1')
        branch5x5 = self.conv_bn_layer(branch5x5, 64, 5, padding=2, name=name+'.branch5x5_2')

        branch3x3dbl = self.conv_bn_layer(x, 64, 1, name=name+'.branch3x3dbl_1')
        branch3x3dbl = self.conv_bn_layer(branch3x3dbl, 96, 3, padding=1, name=name+'.branch3x3dbl_2')
        branch3x3dbl = self.conv_bn_layer(branch3x3dbl, 96, 3, padding=1, name=name+'.branch3x3dbl_3')

        branch_pool = fluid.layers.pool2d(x, pool_size=3, pool_stride=1, pool_padding=1, pool_type='avg')
        branch_pool = self.conv_bn_layer(branch_pool, pool_features, 1, name=name+'.branch_pool')

        return fluid.layers.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=1)
       

    def inceptionB(self, x, name=None):
        branch3x3 = self.conv_bn_layer(x, 384, 3, stride=2, name=name+'.branch3x3')
 
        branch3x3dbl = self.conv_bn_layer(x, 64, 1, name=name+'.branch3x3dbl_1')
        branch3x3dbl = self.conv_bn_layer(branch3x3dbl, 96, 3, padding=1, name=name+'.branch3x3dbl_2')
        branch3x3dbl = self.conv_bn_layer(branch3x3dbl, 96, 3, stride=2, name=name+'.branch3x3dbl_3')

        branch_pool = fluid.layers.pool2d(x, pool_size=3, pool_stride=2, pool_type='max')

        return fluid.layers.concat([branch3x3, branch3x3dbl, branch_pool], axis=1)

    def inceptionC(self, x, c7, name=None):
        branch1x1 = self.conv_bn_layer(x, 192, 1, name=name+'.branch1x1')

        branch7x7 = self.conv_bn_layer(x, c7, 1, name=name+'.branch7x7_1')
        branch7x7 = self.conv_bn_layer(branch7x7, c7, (1, 7), padding=(0, 3), name=name+'.branch7x7_2')
        branch7x7 = self.conv_bn_layer(branch7x7, 192, (7, 1), padding=(3, 0), name=name+'.branch7x7_3')

        branch7x7dbl = self.conv_bn_layer(x, c7, 1, name=name+'.branch7x7dbl_1')
        branch7x7dbl = self.conv_bn_layer(branch7x7dbl, c7, (7, 1), padding=(3, 0), name=name+'.branch7x7dbl_2')
        branch7x7dbl = self.conv_bn_layer(branch7x7dbl, c7, (1, 7), padding=(0, 3), name=name+'.branch7x7dbl_3')
        branch7x7dbl = self.conv_bn_layer(branch7x7dbl, c7, (7, 1), padding=(3, 0), name=name+'.branch7x7dbl_4')
        branch7x7dbl = self.conv_bn_layer(branch7x7dbl, 192, (1, 7), padding=(0, 3), name=name+'.branch7x7dbl_5')

        branch_pool = fluid.layers.pool2d(x, pool_size=3, pool_stride=1, pool_padding=1, pool_type='avg')
        branch_pool = self.conv_bn_layer(branch_pool, 192, 1, name=name+'.branch_pool')

        return fluid.layers.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=1)

    def inceptionD(self, x, name=None):
        branch3x3 = self.conv_bn_layer(x, 192, 1, name=name+'.branch3x3_1')
        branch3x3 = self.conv_bn_layer(branch3x3, 320, 3, stride=2, name=name+'.branch3x3_2')

        branch7x7x3 = self.conv_bn_layer(x, 192, 1, name=name+'.branch7x7x3_1')
        branch7x7x3 = self.conv_bn_layer(branch7x7x3, 192, (1, 7), padding=(0, 3), name=name+'.branch7x7x3_2')
        branch7x7x3 = self.conv_bn_layer(branch7x7x3, 192, (7, 1), padding=(3, 0), name=name+'.branch7x7x3_3')
        branch7x7x3 = self.conv_bn_layer(branch7x7x3, 192, 3, stride=2, name=name+'.branch7x7x3_4')

        branch_pool = fluid.layers.pool2d(x, pool_size=3, pool_stride=2, pool_type='max')
        return fluid.layers.concat([branch3x3, branch7x7x3, branch_pool], axis=1)

    def inceptionE(self, x, name=None):
        branch1x1 = self.conv_bn_layer(x, 320, 1, name=name+'.branch1x1')

        branch3x3 = self.conv_bn_layer(x, 384, 1, name=name+'.branch3x3_1')
        branch3x3_2a = self.conv_bn_layer(branch3x3, 384, (1, 3), padding=(0, 1), name=name+'.branch3x3_2a')
        branch3x3_2b = self.conv_bn_layer(branch3x3, 384, (3, 1), padding=(1, 0), name=name+'.branch3x3_2b')
        branch3x3 = fluid.layers.concat([branch3x3_2a, branch3x3_2b], axis=1)

        branch3x3dbl = self.conv_bn_layer(x, 448, 1, name=name+'.branch3x3dbl_1')
        branch3x3dbl = self.conv_bn_layer(branch3x3dbl, 384, 3, padding=1, name=name+'.branch3x3dbl_2')
        branch3x3dbl_3a = self.conv_bn_layer(branch3x3dbl, 384, (1, 3), padding=(0, 1), name=name+'.branch3x3dbl_3a')
        branch3x3dbl_3b = self.conv_bn_layer(branch3x3dbl, 384, (3, 1), padding=(1, 0), name=name+'.branch3x3dbl_3b')
        branch3x3dbl = fluid.layers.concat([branch3x3dbl_3a, branch3x3dbl_3b], axis=1)

        branch_pool = fluid.layers.pool2d(x, pool_size=3, pool_stride=1, pool_padding=1, pool_type='avg')
        branch_pool = self.conv_bn_layer(branch_pool, 192, 1, name=name+'.branch_pool')

        return fluid.layers.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=1)

    def inceptionAux(self, x, num_classes, name=None):
        x = fluid.layers.pool2d(x, pool_size=5, pool_stride=3, pool_type='avg')
        x = self.conv_bn_layer(x, 128, 1, name=name+'.conv0')
        x = self.conv_bn_layer(x, 768, 5, name=name+'.conv1')
        x = fluid.layers.pool2d(x, global_pooling=True, pool_type='avg')
        x = fluid.layers.flatten(x, axis=1)
        x = fluid.layers.fc(x, size=num_classes)
        return x


    def fid_inceptionA(self, x, pool_features, name=None):
        """ FID block in inception v3
        """
        branch1x1 = self.conv_bn_layer(x, 64, 1, name=name+'.branch1x1')

        branch5x5 = self.conv_bn_layer(x, 48, 1, name=name+'.branch5x5_1')
        branch5x5 = self.conv_bn_layer(branch5x5, 64, 5, padding=2, name=name+'.branch5x5_2')

        branch3x3dbl = self.conv_bn_layer(x, 64, 1, name=name+'.branch3x3dbl_1')
        branch3x3dbl = self.conv_bn_layer(branch3x3dbl, 96, 3, padding=1, name=name+'.branch3x3dbl_2')
        branch3x3dbl = self.conv_bn_layer(branch3x3dbl, 96, 3, padding=1, name=name+'.branch3x3dbl_3')

        branch_pool = fluid.layers.pool2d(x, pool_size=3, pool_stride=1, pool_padding=1, exclusive=True, pool_type='avg')
        branch_pool = self.conv_bn_layer(branch_pool, pool_features, 1, name=name+'.branch_pool')

        return fluid.layers.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=1)

    def fid_inceptionC(self, x, c7, name=None):
        """ FID block in inception v3
        """
        branch1x1 = self.conv_bn_layer(x, 192, 1, name=name+'.branch1x1')

        branch7x7 = self.conv_bn_layer(x, c7, 1, name=name+'.branch7x7_1')
        branch7x7 = self.conv_bn_layer(branch7x7, c7, (1, 7), padding=(0, 3), name=name+'.branch7x7_2')
        branch7x7 = self.conv_bn_layer(branch7x7, 192, (7, 1), padding=(3, 0), name=name+'.branch7x7_3')

        branch7x7dbl = self.conv_bn_layer(x, c7, 1, name=name+'.branch7x7dbl_1')
        branch7x7dbl = self.conv_bn_layer(branch7x7dbl, c7, (7, 1), padding=(3, 0), name=name+'.branch7x7dbl_2')
        branch7x7dbl = self.conv_bn_layer(branch7x7dbl, c7, (1, 7), padding=(0, 3), name=name+'.branch7x7dbl_3')
        branch7x7dbl = self.conv_bn_layer(branch7x7dbl, c7, (7, 1), padding=(3, 0), name=name+'.branch7x7dbl_4')
        branch7x7dbl = self.conv_bn_layer(branch7x7dbl, 192, (1, 7), padding=(0, 3), name=name+'.branch7x7dbl_5')

        branch_pool = fluid.layers.pool2d(x, pool_size=3, pool_stride=1, pool_padding=1, exclusive=True, pool_type='avg')
        branch_pool = self.conv_bn_layer(branch_pool, 192, 1, name=name+'.branch_pool')

        return fluid.layers.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=1)

    def fid_inceptionE_1(self, x, name=None):
        """ FID block in inception v3
        """
        branch1x1 = self.conv_bn_layer(x, 320, 1, name=name+'.branch1x1')

        branch3x3 = self.conv_bn_layer(x, 384, 1, name=name+'.branch3x3_1')
        branch3x3_2a = self.conv_bn_layer(branch3x3, 384, (1, 3), padding=(0, 1), name=name+'.branch3x3_2a')
        branch3x3_2b = self.conv_bn_layer(branch3x3, 384, (3, 1), padding=(1, 0), name=name+'.branch3x3_2b')
        branch3x3 = fluid.layers.concat([branch3x3_2a, branch3x3_2b], axis=1)

        branch3x3dbl = self.conv_bn_layer(x, 448, 1, name=name+'.branch3x3dbl_1')
        branch3x3dbl = self.conv_bn_layer(branch3x3dbl, 384, 3, padding=1, name=name+'.branch3x3dbl_2')
        branch3x3dbl_3a = self.conv_bn_layer(branch3x3dbl, 384, (1, 3), padding=(0, 1), name=name+'.branch3x3dbl_3a')
        branch3x3dbl_3b = self.conv_bn_layer(branch3x3dbl, 384, (3, 1), padding=(1, 0), name=name+'.branch3x3dbl_3b')
        branch3x3dbl = fluid.layers.concat([branch3x3dbl_3a, branch3x3dbl_3b], axis=1)

        branch_pool = fluid.layers.pool2d(x, pool_size=3, pool_stride=1, pool_padding=1, exclusive=True, pool_type='avg')
        branch_pool = self.conv_bn_layer(branch_pool, 192, 1, name=name+'.branch_pool')

        return fluid.layers.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=1)

    def fid_inceptionE_2(self, x, name=None):
        """ FID block in inception v3
        """
        branch1x1 = self.conv_bn_layer(x, 320, 1, name=name+'.branch1x1')

        branch3x3 = self.conv_bn_layer(x, 384, 1, name=name+'.branch3x3_1')
        branch3x3_2a = self.conv_bn_layer(branch3x3, 384, (1, 3), padding=(0, 1), name=name+'.branch3x3_2a')
        branch3x3_2b = self.conv_bn_layer(branch3x3, 384, (3, 1), padding=(1, 0), name=name+'.branch3x3_2b')
        branch3x3 = fluid.layers.concat([branch3x3_2a, branch3x3_2b], axis=1)

        branch3x3dbl = self.conv_bn_layer(x, 448, 1, name=name+'.branch3x3dbl_1')
        branch3x3dbl = self.conv_bn_layer(branch3x3dbl, 384, 3, padding=1, name=name+'.branch3x3dbl_2')
        branch3x3dbl_3a = self.conv_bn_layer(branch3x3dbl, 384, (1, 3), padding=(0, 1), name=name+'.branch3x3dbl_3a')
        branch3x3dbl_3b = self.conv_bn_layer(branch3x3dbl, 384, (3, 1), padding=(1, 0), name=name+'.branch3x3dbl_3b')
        branch3x3dbl = fluid.layers.concat([branch3x3dbl_3a, branch3x3dbl_3b], axis=1)
        
        ### same with paper
        branch_pool = fluid.layers.pool2d(x, pool_size=3, pool_stride=1, pool_padding=1, pool_type='max')
        branch_pool = self.conv_bn_layer(branch_pool, 192, 1, name=name+'.branch_pool')

        return fluid.layers.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=1)

    def conv_bn_layer(self,
                      data,
                      num_filters,
                      filter_size,
                      stride=1,
                      padding=0,
                      groups=1,
                      act='relu',
                      name=None):
        conv = fluid.layers.conv2d(
            input=data,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + ".conv.weight"),
            bias_attr=False,
            name=name)
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            epsilon=0.001,
            name=name+'.bn',
            param_attr=ParamAttr(name=name + ".bn.weight"),
            bias_attr=ParamAttr(name=name + ".bn.bias"),
            moving_mean_name=name + '.bn.running_mean',
            moving_variance_name=name + '.bn.running_var')
