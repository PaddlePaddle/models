# -*- coding=utf-8 -*-
"""
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""

import math
import argparse
import numpy as np
import paddle.fluid as fluid
import paddle.tensor as tensor
from utils import tdm_sampler_prepare, tdm_child_prepare, trace_var
from train_network import DnnLayerClassifierNet


class TdmInferNet(object):
    def __init__(self, args):
        self.input_embed_size = args.query_emb_size
        self.node_embed_size = args.node_emb_size
        self.label_nums = 2
        self.node_nums = args.node_nums
        self.max_layers = args.layer_size
        self.batch_size = args.batch_size
        self.topK = args.topK
        self.child_nums = args.child_nums

        self.layer_list = self.get_layer_list(args)
        self.first_layer_idx = 0
        self.first_layer_node = self.create_first_layer()
        self.layer_classifier = DnnLayerClassifierNet(args)

    def input_data(self):
        input_emb = fluid.layers.data(
            name="input_emb",
            shape=[self.input_embed_size],
            dtype="float32",
        )

        first_layer = fluid.layers.data(
            name="first_layer_node",
            shape=[1],
            dtype="int64",
            lod_level=1,
        )
        first_layer_mask = fluid.layers.data(
            name="first_layer_node_mask",
            shape=[1],
            dtype="int64",
            lod_level=1,
        )

        inputs = [input_emb] + [first_layer] + [first_layer_mask]
        return inputs

    def get_layer_list(self, args):
        """get layer list from tree_emb.txt"""
        layer_list = []
        with open(args.tree_layer_init_path, 'r') as fin:
            for line in fin.readlines():
                l = []
                layer = (line.split('\n'))[0].split(',')
                for node in layer:
                    if node:
                        l.append(node)
                layer_list.append(l)
        return layer_list

    def create_first_layer(self):
        """decide which layer to start"""
        first_layer_id = 0
        for idx, layer in enumerate(self.layer_list):
            if len(layer) > self.topK:
                first_layer_id = idx
                break
        first_layer_node = self.layer_list[first_layer_id]
        self.first_layer_idx = first_layer_id
        return first_layer_node

    def infer_net(self, inputs):
        input_emb = inputs[0]
        current_layer_node = inputs[1]
        current_layer_child_mask = inputs[2]

        node_score = []
        node_list = []

        input_trans_emb = fluid.layers.fc(
            input=input_emb,
            size=self.node_embed_size,
            act=None,
            param_attr=fluid.ParamAttr(name="trans.input_fc.weight"),
            bias_attr=fluid.ParamAttr(name="trans.input_fc.bias"),
        )

        for layer_idx in range(self.first_layer_idx, self.max_layers):
            if layer_idx == 0:
                current_layer_node_num = len(self.first_layer_node)
            else:
                current_layer_node_num = current_layer_node.shape[1] * \
                    current_layer_node.shape[2]

            current_layer_node = fluid.layers.reshape(
                current_layer_node, [self.batch_size, current_layer_node_num])
            current_layer_child_mask = fluid.layers.reshape(
                current_layer_child_mask, [self.batch_size, current_layer_node_num])

            node_emb = fluid.embedding(
                input=current_layer_node,
                size=[self.node_nums, self.node_embed_size],
                param_attr=fluid.ParamAttr(name="tdm.node_emb.weight"))

            layer_classifier_res = self.layer_classifier.classifier_layer_infer(input_trans_emb,
                                                                                node_emb,
                                                                                layer_idx)

            tdm_fc = fluid.layers.fc(input=layer_classifier_res,
                                     size=self.label_nums,
                                     act=None,
                                     num_flatten_dims=2,
                                     param_attr=fluid.ParamAttr(
                                         name="tdm.cls_fc.weight"),
                                     bias_attr=fluid.ParamAttr(name="tdm.cls_fc.bias"))

            prob = fluid.layers.softmax(tdm_fc)
            positive_prob = fluid.layers.slice(
                prob, axes=[2], starts=[1], ends=[2])
            prob_re = fluid.layers.reshape(
                positive_prob, [self.batch_size, current_layer_node_num])

            k = self.topK
            if current_layer_node_num < self.topK:
                k = current_layer_node_num
            _, topk_i = fluid.layers.topk(prob_re, k)  # (None, K)

            top_node = tensor.index_sample(current_layer_node, topk_i)
            prob_re_mask = prob_re * current_layer_child_mask
            topk_value = tensor.index_sample(prob_re_mask, topk_i)
            node_score.append(topk_value)
            node_list.append(top_node)

            if layer_idx < self.max_layers - 1:
                current_layer_node, current_layer_child_mask = \
                    fluid.contrib.layers.tdm_child(x=top_node,
                                                   node_nums=self.node_nums,
                                                   child_nums=self.child_nums,
                                                   param_attr=fluid.ParamAttr(name="TDM_Tree_Info"), dtype='int64')

        total_node_score = fluid.layers.concat(node_score, axis=1)
        total_node = fluid.layers.concat(node_list, axis=1)

        _, res_i = fluid.layers.topk(total_node_score, self.topK)
        res_layer_node = tensor.index_sample(total_node, res_i)
        res_node = fluid.layers.reshape(res_layer_node, [-1, self.topK, 1])

        tree_info = fluid.default_main_program().global_block().var("TDM_Tree_Info")
        res_node_emb = fluid.layers.gather_nd(tree_info, res_node)

        res_item = fluid.layers.slice(
            res_node_emb, axes=[2], starts=[0], ends=[1])
        res_item_re = fluid.layers.reshape(res_item, [-1, self.topK])
        return res_item_re
