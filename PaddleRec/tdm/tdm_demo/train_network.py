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
from utils import tdm_sampler_prepare, tdm_child_prepare, trace_var


class TdmTrainNet(object):

    def __init__(self, args):
        self.input_embed_size = args.query_emb_size
        self.node_embed_size = args.node_emb_size
        self.label_nums = 2
        self.node_nums = args.node_nums
        self.max_layers = args.layer_size
        self.neg_sampling_list = args.neg_sampling_list
        self.output_positive = True
        self.need_trace = args.need_trace
        self.need_detail = args.need_detail

        if not args.load_model:
            self.tdm_sampler_prepare_dict = tdm_sampler_prepare(args)
            print("--Layer node num list--: {}".format(
                self.tdm_sampler_prepare_dict['layer_node_num_list']))
            self.layer_node_num_list = self.tdm_sampler_prepare_dict['layer_node_num_list']

            print("--leaf node num--: {}".format(
                self.tdm_sampler_prepare_dict['leaf_node_num']))
            self.leaf_node_num = self.tdm_sampler_prepare_dict['leaf_node_num']
            self.info_array = tdm_child_prepare(args)
        else:
            self.layer_node_num_list = args.layer_node_num_list
            self.leaf_node_num = args.leaf_node_num

        self.get_tree_info(args)
        self.input_trans_layer = InputTransNet(args)
        self.layer_classifier = DnnLayerClassifierNet(args)

    def get_tree_info(self, args):
        fluid.default_startup_program().global_block().create_var(
            name="TDM_Tree_Info", dtype=fluid.core.VarDesc.VarType.INT32, shape=[args.node_nums, 3 + args.child_nums],
            persistable=True,
            initializer=fluid.initializer.ConstantInitializer(0))
        tdm_tree_info = fluid.default_main_program().global_block().create_var(
            name="TDM_Tree_Info", dtype=fluid.core.VarDesc.VarType.INT32, shape=[args.node_nums, 3 + args.child_nums],
            persistable=True)

    def input_data(self):
        input_emb = fluid.data(
            name="input_emb",
            shape=[None, self.input_embed_size],
            dtype="float32",
        )

        item_label = fluid.data(
            name="item_label",
            shape=[None, 1],
            dtype="int64",
        )

        inputs = [input_emb] + [item_label]
        return inputs

    def tdm(self, inputs):
        input_emb = inputs[0]
        item_label = inputs[1]
        trace_var(input_emb, "[TDM][inputs]", "input_emb",
                  self.need_trace, self.need_detail)
        trace_var(item_label, "[TDM][inputs]",
                  "item_label", self.need_trace, self.need_detail)

        sample_nodes, sample_label, sample_mask = fluid.contrib.layers.tdm_sampler(
            x=item_label,
            neg_samples_num_list=self.neg_sampling_list,
            layer_node_num_list=self.layer_node_num_list,
            leaf_node_num=self.leaf_node_num,
            tree_travel_attr=fluid.ParamAttr(name="TDM_Tree_Travel"),
            tree_layer_attr=fluid.ParamAttr(name="TDM_Tree_Layer"),
            output_positive=self.output_positive,
            output_list=True,
            seed=0,
            dtype='int64'
        )

        trace_var(sample_nodes, "[TDM][tdm_sample]",
                  "sample_nodes", self.need_trace, self.need_detail)
        trace_var(sample_label, "[TDM][tdm_sample]",
                  "sample_label", self.need_trace, self.need_detail)
        trace_var(sample_mask, "[TDM][tdm_sample]",
                  "sample_mask", self.need_trace, self.need_detail)

        sample_nodes_emb = [
            fluid.embedding(
                input=sample_nodes[i],
                is_sparse=True,
                size=[self.node_nums, self.node_embed_size],
                param_attr=fluid.ParamAttr(
                    name="tdm.node_emb.weight")
            ) for i in range(self.max_layers)
        ]
        sample_nodes_emb = [
            fluid.layers.reshape(sample_nodes_emb[i],
                                 [-1, self.neg_sampling_list[i] +
                                     self.output_positive, self.node_embed_size]
                                 ) for i in range(self.max_layers)
        ]
        trace_var(sample_nodes_emb, "[TDM][tdm_sample]",
                  "sample_nodes_emb", self.need_trace, self.need_detail)

        input_trans_emb = self.input_trans_layer.input_trans_layer(input_emb)
        trace_var(input_trans_emb, "[TDM][input_trans]",
                  "input_trans_emb", self.need_trace, self.need_detail)

        layer_classifier_res = self.layer_classifier.classifier_layer(
            input_trans_emb, sample_nodes_emb)
        trace_var(layer_classifier_res, "[TDM][classifier_layer]",
                  "layer_classifier_res", self.need_trace, self.need_detail)

        tdm_fc = fluid.layers.fc(input=layer_classifier_res,
                                 size=self.label_nums,
                                 act=None,
                                 num_flatten_dims=2,
                                 param_attr=fluid.ParamAttr(
                                     name="tdm.cls_fc.weight"),
                                 bias_attr=fluid.ParamAttr(name="tdm.cls_fc.bias"))
        trace_var(tdm_fc, "[TDM][cls_fc]", "tdm_fc",
                  self.need_trace, self.need_detail)

        tdm_fc_re = fluid.layers.reshape(tdm_fc, [-1, 2])
        sample_label = fluid.layers.concat(sample_label, axis=1)
        labels_reshape = fluid.layers.reshape(sample_label, [-1, 1])
        cost, softmax_prob = fluid.layers.softmax_with_cross_entropy(
            logits=tdm_fc_re, label=labels_reshape, return_softmax=True)

        sample_mask = fluid.layers.concat(sample_mask, axis=1)
        mask_reshape = fluid.layers.reshape(sample_mask, [-1, 1])
        mask_index = fluid.layers.where(mask_reshape != 0)
        mask_cost = fluid.layers.gather_nd(cost, mask_index)
        avg_cost = fluid.layers.reduce_mean(mask_cost)
        acc = fluid.layers.accuracy(input=softmax_prob, label=labels_reshape)
        return avg_cost, acc


class InputTransNet(object):
    def __init__(self, args):
        self.node_embed_size = args.node_emb_size
        self.max_layers = args.layer_size
        self.is_test = args.is_test

    def input_trans_layer(self, input_emb):
        input_fc_out = fluid.layers.fc(
            input=input_emb,
            size=self.node_embed_size,
            act=None,
            param_attr=fluid.ParamAttr(name="trans.input_fc.weight"),
            bias_attr=fluid.ParamAttr(name="trans.input_fc.bias"),
        )

        input_layer_fc_out = [
            fluid.layers.fc(
                input=input_fc_out,
                size=self.node_embed_size,
                act="tanh",
                param_attr=fluid.ParamAttr(
                    name="trans.layer_fc.weight." + str(i)),
                bias_attr=fluid.ParamAttr(name="trans.layer_fc.bias."+str(i)),
            ) for i in range(self.max_layers)
        ]

        return input_layer_fc_out


class DnnLayerClassifierNet(object):
    def __init__(self, args):
        self.node_embed_size = 64
        self.max_layers = args.layer_size
        self.neg_sampling_list = args.neg_sampling_list
        self.output_positive = True
        self.is_test = args.is_test
        self.child_nums = args.child_nums

    def _expand_layer(self, input_layer, node, layer_idx):
        input_layer_unsequeeze = fluid.layers.unsqueeze(
            input=input_layer, axes=[1])
        if self.is_test:
            input_layer_expand = fluid.layers.expand(
                input_layer_unsequeeze, expand_times=[1, node.shape[1], 1])
        else:
            input_layer_expand = fluid.layers.expand(
                input_layer_unsequeeze, expand_times=[1, node[layer_idx].shape[1], 1])
        return input_layer_expand

    def classifier_layer(self, input, node):

        input_expand = [
            self._expand_layer(input[i], node, i) for i in range(self.max_layers)
        ]

        input_node_concat = [
            fluid.layers.concat(
                input=[input_expand[i], node[i]],
                axis=2) for i in range(self.max_layers)
        ]
        hidden_states_fc = [
            fluid.layers.fc(
                input=input_node_concat[i],
                size=self.node_embed_size,
                num_flatten_dims=2,
                act="tanh",
                param_attr=fluid.ParamAttr(
                    name="cls.concat_fc.weight."+str(i)),
                bias_attr=fluid.ParamAttr(name="cls.concat_fc.bias."+str(i))
            ) for i in range(self.max_layers)
        ]
        hidden_states_concat = fluid.layers.concat(hidden_states_fc, axis=1)
        return hidden_states_concat

    def classifier_layer_infer(self, input, node, layer_idx):
        input_expand = self._expand_layer(input, node, layer_idx)

        input_node_concat = fluid.layers.concat(
            input=[input_expand, node], axis=2)

        hidden_states_fc = fluid.layers.fc(
            input=input_node_concat,
            size=self.node_embed_size,
            num_flatten_dims=2,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="cls.concat_fc.weight."+str(layer_idx)),
            bias_attr=fluid.ParamAttr(name="cls.concat_fc.bias."+str(layer_idx)))
        return hidden_states_fc
