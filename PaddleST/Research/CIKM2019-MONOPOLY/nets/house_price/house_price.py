#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################


"""
File: nets/house_price/house_price.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import math
import numpy as np

import paddle.fluid as fluid

from nets.base_net import BaseNet
from datasets.house_price.baseline_sklearn import CityInfo


class HousePrice(BaseNet):
    """
    net class: construct net
    """
    def __init__(self, FLAGS):
        super(HousePrice, self).__init__(FLAGS)
        self.city_info = CityInfo(FLAGS.city_name)
        
    def emb_lookup_fn(self, input, dict_dim, emb_dim, layer_name, FLAGS,
            padding_idx=None, init_val=0.0):
        """
        get embedding out with params
        """
        output = fluid.layers.embedding(
            input=input,
            size=[dict_dim, emb_dim],
            padding_idx=padding_idx,
            param_attr=fluid.ParamAttr(
                name=layer_name,
                initializer=fluid.initializer.ConstantInitializer(init_val)),
                is_sparse=True)
        return output
 
    def fc_fn(self, input, output_size, act, layer_name, FLAGS, num_flatten_dims=1):
        """
        pack fc op
        """
        dev = 1.0 / math.sqrt(output_size)
        _fc = fluid.layers.fc(
            input=input,
            size=output_size,
            num_flatten_dims=num_flatten_dims,
            param_attr=fluid.ParamAttr(
                name=layer_name + "_fc_w",
                initializer=fluid.initializer.Xavier(uniform=False)),
                #initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=dev)),
            bias_attr=fluid.ParamAttr(
                name=layer_name + "_fc_bias",
                initializer=fluid.initializer.Constant(value=0.0)),
            act=act)
        return _fc
 
    def pred_format(self, result, **kwargs):
        """
            format pred output
        """
        if result is None or result in ['_PRE_']:
            return

        def _softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        if result == '_POST_':
            h_attr_w = fluid.global_scope().find_var("house_self_fc_w").get_tensor()
            h_attr_b = fluid.global_scope().find_var("house_self_fc_bias").get_tensor()
            dis_w = fluid.global_scope().find_var("dis_w").get_tensor()
            bids = fluid.global_scope().find_var("bids").get_tensor()
            print("h_attr_w: %s" % (" ".join(map(str, _softmax(np.array(h_attr_w).flatten())))))
            print("h_attr_b: %s" % (" ".join(map(str, np.array(h_attr_b)))))
            print("dis_w: %s" % (" ".join(map(str, _softmax(np.array(np.mean(dis_w, 0)))))))
            print("bids: %s" % (" ".join(map(str, np.array(bids).flatten()))))
            return

        label = np.array(result[0]).T.flatten().tolist()
        pred = np.array(result[1]).T.flatten().tolist()
        for i in range(len(pred)):
            print("qid\t%s\t%s" % (label[i], pred[i]))

    def net(self, inputs):
        """
        user-defined interface
        """
        """
            feature: dict. {"label": xxx, "ct_onehot": xxxx,,...}
        """
        FLAGS = self._flags

        label = inputs['label']
        public_bids = inputs['public_bid']

        max_house_num = FLAGS.max_house_num
        max_public_num = FLAGS.max_public_num
        pred_keys = inputs.keys() 
        #step1. get house self feature
        if FLAGS.with_house_attr:
            def _get_house_attr(name, attr_vec_size):
                h_onehot = fluid.layers.reshape(inputs[name], [-1, attr_vec_size])
                h_attr = self.fc_fn(h_onehot, 1, act=None, layer_name=name, FLAGS=FLAGS)
                return h_attr
         
            house_business = _get_house_attr("house_business", self.city_info.business_num)
            house_wuye = _get_house_attr("house_wuye", self.city_info.wuye_num)
            house_kfs = _get_house_attr("house_kfs", self.city_info.kfs_num)
            house_age = _get_house_attr("house_age", self.city_info.age_num)
            house_lou = _get_house_attr("house_lou", self.city_info.lou_num)
            
            house_vec = fluid.layers.concat([house_business, house_wuye, house_kfs, house_age, house_lou], 1)
        else:
            #no house attr
            house_vec = fluid.layers.reshape(inputs["house_business"], [-1, self.city_info.business_num])
            pred_keys.remove('house_wuye')
            pred_keys.remove('house_kfs')
            pred_keys.remove('house_age')
            pred_keys.remove('house_lou')

        house_self = self.fc_fn(house_vec, 1, act='sigmoid', layer_name='house_self', FLAGS=FLAGS)
        house_self = fluid.layers.reshape(house_self, [-1, 1])
       
        #step2. get nearby house and public poi feature
        #public poi embeddings matrix
        bid_embed = self.emb_lookup_fn(public_bids, self.city_info.public_num, 1, 'bids', FLAGS, None,
                self.city_info.average_price)
       
        dis_dim = 1 #only line dis
        if FLAGS.with_car_dis:
            dis_dim = 2 #add car drive dis

        #nearby house and public poi distance weight matrix
        dis_w = fluid.layers.create_parameter(shape=[max_house_num + max_public_num, dis_dim],
                dtype='float32', name='dis_w') 
        house_price = inputs['house_price']
        public_price = fluid.layers.reshape(bid_embed, [-1, max_public_num])
        #nearby price
        price_vec = fluid.layers.concat([house_price, public_price], 1)
       
        #nearby price weight
        house_dis = fluid.layers.reshape(inputs['house_dis'], [-1, max_house_num, dis_dim])
        public_dis = fluid.layers.reshape(inputs['public_dis'], [-1, max_public_num, dis_dim])
        dis_vec = fluid.layers.concat([house_dis, public_dis], 1)
        dis_w = fluid.layers.reshape(dis_w, [max_house_num + max_public_num, dis_dim])
        dis_vec = fluid.layers.reduce_sum(dis_vec * dis_w, 2) 
        house_mask = fluid.layers.sequence_mask(fluid.layers.reshape(inputs['house_num'], [-1]),
                max_house_num) #remove padded
        public_mask = fluid.layers.sequence_mask(fluid.layers.reshape(inputs['public_num'], [-1]),
                max_public_num) #remove padded
        combine_mask =  fluid.layers.cast(x=fluid.layers.concat([house_mask, public_mask], 1),
                dtype="float32")
        adder = (1.0 - combine_mask) * -10000.0
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        dis_vec += adder
        price_weight = fluid.layers.softmax(dis_vec)
        
        combine_price = price_vec * price_weight
        
        #step3. merge house_self and nearby house and public price: [-1, 1] * [-1, 1] 
        pred = house_self * fluid.layers.unsqueeze(fluid.layers.reduce_sum(combine_price, 1), [1])
        #fluid.layers.Print(pred, message=None, summarize=-1)
        #fluid.layers.Print(label, message=None, summarize=-1)
        
        loss = fluid.layers.square_error_cost(input=pred, label=label)

        avg_cost = fluid.layers.mean(loss)

        # debug output info during training
        debug_output = {}
        model_output = {}
        net_output = {"debug_output": debug_output, 
                      "model_output": model_output}

        model_output['feeded_var_names'] = pred_keys   
        model_output['fetch_targets'] = [label, pred]
        model_output['loss'] = avg_cost

        #debug_output['pred'] = pred 
        debug_output['loss'] = avg_cost
        #debug_output['label'] = label
        #debug_output['public_bids'] = public_bids
        return net_output

