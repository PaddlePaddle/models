#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import os
import time
import sys
import paddle.fluid as fluid


class TALL(object):
    def __init__(self, mode, cfg):
	self.images = cfg["images"]
	self.sentences = cfg["sentences"]
	self.semantic_size = cfg["semantic_size"]
	self.hidden_size = cfg["hidden_size"]
	self.output_size = cfg["output_size"]

    def _cross_modal_comb(visual_feat, sentence_embed):
        #batch_size = visual_feat.size(0)
        visual_feat = fluid.layers.reshape(visual_feat, [1, -1, semantic_size])
        vv_feature = fluid.layers.expand(visual_feat, [train_batch_size, 1, 1])
        sentence_embed = fluid.layers.reshape(sentence_embed, [-1, 1, semantic_size])
        ss_feature = fluid.layers.expand(sentence_embed, [1, train_batch_size, 1])

        concat_feature = fluid.layers.concat([vv_feature, ss_feature], axis = 2) #1,1,2048

        mul_feature = vv_feature * ss_feature # B,B,1024
        add_feature = vv_feature + ss_feature # B,B,1024

        comb_feature = fluid.layers.concat([mul_feature, add_feature, concat_feature], axis = 2)
        return comb_feature


    def net(self)	
       # visual2semantic
    	transformed_clip = fluid.layers.fc(
        	input=self.images,
        	size=self.semantic_size,
        	act=None,
        	name='v2s_lt',
        	param_attr=fluid.ParamAttr(
            		name='v2s_lt_weights',
            		initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=0)),
        	bias_attr=False)
    	#l2_normalize
    	transformed_clip = fluid.layers.l2_normalize(x=transformed_clip, axis=1)
    	# sentence2semantic
   	transformed_sentence = fluid.layers.fc(
            input=self.sentences,
            size=self.semantic_size,
            act=None,
            name='s2s_lt',
            param_attr=fluid.ParamAttr(
            	name='s2s_lt_weights',
            	initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=0)),
            bias_attr=False)
    	#l2_normalize
    	transformed_sentence = fluid.layers.l2_normalize(x=transformed_sentence, axis=1)
        
	cross_modal_vec=_cross_modal_comb(transformed_clip_train, transformed_sentence)
    	cross_modal_vec=fluid.layers.unsqueeze(input=cross_modal_vec, axes=[0])
    	cross_modal_vec=fluid.layers.transpose(cross_modal_vec, perm=[0, 3, 1, 2])
    
    	mid_output = fluid.layers.conv2d(
            input=cross_modal_vec,
            num_filters=self.hidden_size,
            filter_size=1,
            stride=1,
            act="relu",
            param_attr=fluid.param_attr.ParamAttr(name="mid_out_weights"),
            bias_attr=False)

    	sim_score_mat = fluid.layers.conv2d(
            input=mid_output,
            num_filters=self.output_size,
            filter_size=1,
            stride=1,
            act=None,
            param_attr=fluid.param_attr.ParamAttr(name="sim_mat_weights"),
            bias_attr=False)

    	sim_score_mat = fluid.layers.squeeze(input=sim_score_mat, axes=[0])
    	return sim_score_mat	
