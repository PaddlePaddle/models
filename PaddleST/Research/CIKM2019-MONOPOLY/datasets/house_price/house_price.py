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
File: house_price.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import random
import paddle.fluid as fluid

from datasets.base_dataset import BaseDataset
from datasets.house_price.baseline_sklearn import CityInfo


class HousePrice(BaseDataset):
    """
    shop location dataset 
    """
    def __init__(self, flags):
        super(HousePrice, self).__init__(flags)
        self.city_info = CityInfo(flags.city_name)
        
    def parse_context(self, inputs):
        """
        provide input context
        """

        """
        set inputs_kv: please set key as the same as layer.data.name

        notice:
        (1)
        If user defined "inputs key" is different from layer.data.name,
        the frame will rewrite "inputs key" with layer.data.name
        (2)
        The param "inputs" will be passed to user defined nets class through
        the nets class interface function : net(self, FLAGS, inputs), 
        """
        inputs['label'] = fluid.layers.data(name="label", shape=[1], dtype="float32", lod_level=0)
        #if self._flags.dataset_split_name != 'train':
        #    inputs['qid'] = fluid.layers.data(name='qid', shape=[1], dtype="int32", lod_level=0) 
        #house self feature 
        inputs['house_business'] = fluid.layers.data(name="house_business", shape=[self.city_info.business_num],
                dtype="float32", lod_level=0)
        inputs['house_wuye'] = fluid.layers.data(name="house_wuye", shape=[self.city_info.wuye_num],
                dtype="float32", lod_level=0)
        inputs['house_kfs'] = fluid.layers.data(name="house_kfs", shape=[self.city_info.kfs_num],
                dtype="float32", lod_level=0)
        inputs['house_age'] = fluid.layers.data(name="house_age", shape=[self.city_info.age_num],
                dtype="float32", lod_level=0)
        inputs['house_lou'] = fluid.layers.data(name="house_lou", shape=[self.city_info.lou_num],
                dtype="float32", lod_level=0)
       
        #nearby house and public poi
        inputs['house_price'] = fluid.layers.data(name="house_price", shape=[self._flags.max_house_num],
                dtype="float32", lod_level=0)
        inputs['public_bid'] = fluid.layers.data(name="public_bid", shape=[1],
                dtype="int64", lod_level=1)
        inputs['house_dis'] = fluid.layers.data(name="house_dis", shape=[self._flags.max_house_num * 2],
                dtype="float32", lod_level=0)
        inputs['public_dis'] = fluid.layers.data(name="public_dis", shape=[self._flags.max_public_num * 2],
                dtype="float32", lod_level=0)
        
        inputs['house_num'] = fluid.layers.data(name="house_num", shape=[1], dtype="float32", lod_level=0)
        inputs['public_num'] = fluid.layers.data(name="public_num", shape=[1], dtype="float32", lod_level=0)
                                            
        context = {"inputs": inputs}
        #set debug list, print info during training
        #debug_list = [key for key in inputs]
        #context["debug_list"] = ["label", "house_num"]
        return context
 
    def _normalize_distance_factor(self, dis_vec):
        sum = 0.0
        for d in dis_vec:
            sum += 1.0 / d
        ret = []
        for d in dis_vec:
            ret.append(1.0 / (d * sum))

        return ret

    def parse_oneline(self, line):
        """
        parse sample 
        """
        cols = line.strip('\r\n').split('\t')
        max_house_num = self._flags.max_house_num
        max_public_num = self._flags.max_public_num
        pred = False if self._flags.dataset_split_name == 'train' else True
        
        radius = self._flags.dis_radius
        
        p_info = cols[0].split()
        label = float(p_info[0])
        samples = [('label', [float(label)])]
        
        #house self info
        h_num = int(p_info[1])
        p_num = int(p_info[2])
        onehot_ids = cols[1].split()
        def _get_onehot(idx, num):
            onehot = [0.0] * num
            if idx >= 0 and idx < num:
                onehot[idx] = 1.0
            return onehot
        
        onehot_business = _get_onehot(int(onehot_ids[0]), self.city_info.business_num)
        onehot_wuye = _get_onehot(int(onehot_ids[1]), self.city_info.wuye_num)
        onehot_kfs = _get_onehot(int(onehot_ids[2]), self.city_info.kfs_num)
        onehot_age = _get_onehot(int(onehot_ids[3]), self.city_info.age_num)
        onehot_lou = _get_onehot(int(onehot_ids[4]), self.city_info.lou_num)
        
        #nearby house and public info
        h_p_info = cols[2].split() 
        h_p_dis = cols[3].split()
        h_p_car = []
        if self._flags.with_car_dis:
            h_p_car = cols[4].split()
            assert(len(h_p_car) == len(h_p_dis))

        #if h_num < 1 or p_num < 1:
        #    print("%s, invalid h_num or p_num." % line, file=sys.stderr)
        #    return

        assert(len(h_p_info) == (h_num + p_num) and len(h_p_info) == len(h_p_dis))

        p_id = []
        p_dis = [] 
        h_price = []
        h_dis = []
        for i in range(h_num + p_num):
            if float(h_p_dis[i]) > radius or (len(h_p_car) > 0 and float(h_p_car[i]) < 0):
                continue
            if i < h_num:
                if len(h_price) >= max_house_num:
                    continue
                pinfo = h_p_info[i].split(':')
                #h_price += float(pinfo[1]) * float(h_p_dis[i])
                h_price.append(float(pinfo[1]))
                if len(h_p_car) > 0:
                    h_dis.extend([float(h_p_dis[i]), float(h_p_car[i])])
                else:
                    h_dis.append(float(h_p_dis[i]))
            else:
                if len(p_id) >= max_public_num:
                    break
                p_id.append(int(h_p_info[i]))
                if len(h_p_car) > 0:
                    p_dis.extend([float(h_p_dis[i]), float(h_p_car[i])])
                else:
                    p_dis.append(float(h_p_dis[i]))
        
        qid = 0
        if self._flags.avg_eval:
            if len(h_price) > 0:
                avg_h = np.average(h_price)
                h_dis = self._normalize_distance_factor(h_dis) 
                weight_h = np.sum(np.array(h_price) * h_dis / np.sum(h_dis))
            else:
                avg_h = self.city_info.average_price
                weight_h = self.city_info.average_price
            print("%s\t%s\t%s\t%s\t%s" % (qid, label, avg_h, weight_h, self.city_info.average_price))
            return

        if len(h_price) < 1 and len(p_id) < 1:
            #sys.stderr.write("invalid line.\n")
            return
        h_num = len(h_price)
        p_num = len(p_id)
        #if pred:
        #    samples.append(('qid', [qid]))
       
        samples.append(('house_business', onehot_business)) 
        samples.append(('house_wuye', onehot_wuye)) 
        samples.append(('house_kfs', onehot_kfs))
        samples.append(('house_age', onehot_age))
        samples.append(('house_lou', onehot_lou)) 

        while len(h_price) < max_house_num:
            h_price.append(self.city_info.average_price)
            if len(h_p_car) > 0:
                h_dis.extend([radius, 2 * radius])
            else:
                h_dis.append(radius)
        while len(p_id) < max_public_num:
            p_id.append(0)
            if len(h_p_car) > 0:
                p_dis.extend([radius, 2 * radius])
            else:
                p_dis.append(radius)

        samples.append(('house_price', h_price))
        samples.append(('public_bid', p_id))
        samples.append(('house_dis', h_dis))
        samples.append(('public_dis', p_dis))

        samples.append(('house_num', [h_num])) 
        samples.append(('public_num', [p_num]))
       
        yield samples

