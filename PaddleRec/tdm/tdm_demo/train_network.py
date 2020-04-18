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
import logging
import paddle.fluid as fluid
from utils import tdm_sampler_prepare, tdm_child_prepare, tdm_emb_prepare, trace_var

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


class TdmTrainNet(object):
    """
    TDM-Demo网络的主要流程部分
    """

    def __init__(self, args):
        self.input_embed_size = args.query_emb_size
        self.node_emb_size = args.node_emb_size
        self.label_nums = 2
        self.node_nums = args.node_nums
        self.max_layers = args.layer_size
        self.neg_sampling_list = args.neg_sampling_list
        self.output_positive = True
        self.get_tree_info(args)

        # 设置是否需要进行数值的debug
        self.need_trace = args.need_trace
        self.need_detail = args.need_detail

        if not args.load_model:
            # 每次模型训练仅需运行一次，灌入生成的明文，拿到树结构信息
            # 将其保存为paddle的二进制init_model
            # 下一次调试或训练即可load init_model,快速启动且内存占用更小
            self.tdm_param_prepare_dict = tdm_sampler_prepare(args)
            logger.info("--Layer node num list--: {}".format(
                self.tdm_param_prepare_dict['layer_node_num_list']))
            self.layer_node_num_list = self.tdm_param_prepare_dict['layer_node_num_list']

            logger.info("--Leaf node num--: {}".format(
                self.tdm_param_prepare_dict['leaf_node_num']))
            self.leaf_node_num = self.tdm_param_prepare_dict['leaf_node_num']

            self.tdm_param_prepare_dict['info_array'] = tdm_child_prepare(
                args)
            logger.info(
                "--Tree Info array shape {}--".format(self.tdm_param_prepare_dict['info_array'].shape))

            self.tdm_param_prepare_dict['emb_array'] = tdm_emb_prepare(args)
            logger.info(
                "--Tree Emb array shape {}--".format(self.tdm_param_prepare_dict['emb_array'].shape))
        else:
            self.layer_node_num_list = args.layer_node_num_list
            self.leaf_node_num = args.leaf_node_num

        self.input_trans_layer = InputTransNet(args)
        self.layer_classifier = DnnLayerClassifierNet(args)

    def get_tree_info(self, args):
        """
        TDM_Tree_Info 虽然在训练过程中没有用到，但在预测网络中会使用。
        如果希望保存的模型直接用来预测，不再有额外的生成tree_info参数的步骤，
        则可以在训练组网中添加tree_info参数，训练保存模型时可以进行保存。
        """
        fluid.default_startup_program().global_block().create_var(
            name="TDM_Tree_Info", dtype=fluid.core.VarDesc.VarType.INT32, shape=[args.node_nums, 3 + args.child_nums],
            persistable=True,
            initializer=fluid.initializer.ConstantInitializer(0))
        tdm_tree_info = fluid.default_main_program().global_block().create_var(
            name="TDM_Tree_Info", dtype=fluid.core.VarDesc.VarType.INT32, shape=[args.node_nums, 3 + args.child_nums],
            persistable=True)

    def input_data(self):
        """
        指定tdm训练网络的输入变量
        """
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
        """
        tdm训练网络的主要流程部分
        """
        input_emb = inputs[0]
        item_label = inputs[1]

        # trace_var用于在静态图的调试中打印参数信息细节:
        # 将 need_trace设置为True，可以在日志中看到参数的前向信息（数值默认前20个）
        # 将 need_detail设置为True，可以在日志中看到参数的前向全部数值
        trace_var(input_emb, "[TDM][inputs]", "input_emb",
                  self.need_trace, self.need_detail)
        trace_var(item_label, "[TDM][inputs]",
                  "item_label", self.need_trace, self.need_detail)

        # 根据输入的item的正样本在给定的树上进行负采样
        # sample_nodes 是采样的node_id的结果，包含正负样本
        # sample_label 是采样的node_id对应的正负标签
        # sample_mask 是为了保持tensor维度一致，padding部分的标签，若为0，则是padding的虚拟node_id
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
            tree_dtype='int64',
            dtype='int64'
        )

        trace_var(sample_nodes, "[TDM][tdm_sample]",
                  "sample_nodes", self.need_trace, self.need_detail)
        trace_var(sample_label, "[TDM][tdm_sample]",
                  "sample_label", self.need_trace, self.need_detail)
        trace_var(sample_mask, "[TDM][tdm_sample]",
                  "sample_mask", self.need_trace, self.need_detail)

        # 查表得到每个节点的Embedding
        sample_nodes_emb = [
            fluid.embedding(
                input=sample_nodes[i],
                is_sparse=True,
                size=[self.node_nums, self.node_emb_size],
                param_attr=fluid.ParamAttr(
                    name="TDM_Tree_Emb")
            ) for i in range(self.max_layers)
        ]

        # 此处进行Reshape是为了之后层次化的分类器训练
        sample_nodes_emb = [
            fluid.layers.reshape(sample_nodes_emb[i],
                                 [-1, self.neg_sampling_list[i] +
                                     self.output_positive, self.node_emb_size]
                                 ) for i in range(self.max_layers)
        ]
        trace_var(sample_nodes_emb, "[TDM][tdm_sample]",
                  "sample_nodes_emb", self.need_trace, self.need_detail)

        # 对输入的input_emb进行转换，使其维度与node_emb维度一致
        input_trans_emb = self.input_trans_layer.input_trans_layer(input_emb)
        trace_var(input_trans_emb, "[TDM][input_trans]",
                  "input_trans_emb", self.need_trace, self.need_detail)

        # 分类器的主体网络，分别训练不同层次的分类器
        layer_classifier_res = self.layer_classifier.classifier_layer(
            input_trans_emb, sample_nodes_emb)
        trace_var(layer_classifier_res, "[TDM][classifier_layer]",
                  "layer_classifier_res", self.need_trace, self.need_detail)

        # 最后的概率判别FC，将所有层次的node分类结果放到一起以相同的标准进行判别
        # 考虑到树极大可能不平衡，有些item不在最后一层，所以需要这样的机制保证每个item都有机会被召回
        tdm_fc = fluid.layers.fc(input=layer_classifier_res,
                                 size=self.label_nums,
                                 act=None,
                                 num_flatten_dims=2,
                                 param_attr=fluid.ParamAttr(
                                     name="tdm.cls_fc.weight"),
                                 bias_attr=fluid.ParamAttr(name="tdm.cls_fc.bias"))
        trace_var(tdm_fc, "[TDM][cls_fc]", "tdm_fc",
                  self.need_trace, self.need_detail)

        # 将loss打平，放到一起计算整体网络的loss
        tdm_fc_re = fluid.layers.reshape(tdm_fc, [-1, 2])

        # 若想对各个层次的loss辅以不同的权重，则在此处无需concat
        # 支持各个层次分别计算loss，再乘相应的权重
        sample_label = fluid.layers.concat(sample_label, axis=1)
        labels_reshape = fluid.layers.reshape(sample_label, [-1, 1])

        # 计算整体的loss并得到softmax的输出
        cost, softmax_prob = fluid.layers.softmax_with_cross_entropy(
            logits=tdm_fc_re, label=labels_reshape, return_softmax=True)

        # 通过mask过滤掉虚拟节点的loss
        sample_mask = fluid.layers.concat(sample_mask, axis=1)
        mask_reshape = fluid.layers.reshape(sample_mask, [-1, 1])
        mask_index = fluid.layers.where(mask_reshape != 0)
        mask_cost = fluid.layers.gather_nd(cost, mask_index)

        # 计算该batch的均值loss，同时计算acc, 亦可在这里计算auc
        avg_cost = fluid.layers.reduce_mean(mask_cost)
        acc = fluid.layers.accuracy(input=softmax_prob, label=labels_reshape)
        return avg_cost, acc


class InputTransNet(object):
    """
    输入侧组网的主要部分
    """

    def __init__(self, args):
        self.node_emb_size = args.node_emb_size
        self.max_layers = args.layer_size
        self.is_test = args.is_test

    def input_trans_layer(self, input_emb):
        """
        输入侧训练组网
        """
        # 将input映射到与node相同的维度
        input_fc_out = fluid.layers.fc(
            input=input_emb,
            size=self.node_emb_size,
            act=None,
            param_attr=fluid.ParamAttr(name="trans.input_fc.weight"),
            bias_attr=fluid.ParamAttr(name="trans.input_fc.bias"),
        )

        # 将input_emb映射到各个不同层次的向量表示空间
        input_layer_fc_out = [
            fluid.layers.fc(
                input=input_fc_out,
                size=self.node_emb_size,
                act="tanh",
                param_attr=fluid.ParamAttr(
                    name="trans.layer_fc.weight." + str(i)),
                bias_attr=fluid.ParamAttr(name="trans.layer_fc.bias."+str(i)),
            ) for i in range(self.max_layers)
        ]

        return input_layer_fc_out

    def input_fc_infer(self, input_emb):
        """
        输入侧预测组网第一部分，将input转换为node同维度
        """
        # 组网与训练时保持一致
        input_fc_out = fluid.layers.fc(
            input=input_emb,
            size=self.node_emb_size,
            act=None,
            param_attr=fluid.ParamAttr(name="trans.input_fc.weight"),
            bias_attr=fluid.ParamAttr(name="trans.input_fc.bias"),
        )
        return input_fc_out

    def layer_fc_infer(self, input_fc_out, layer_idx):
        """
        输入侧预测组网第二部分，将input映射到不同层次的向量空间
        """
        # 组网与训练保持一致，通过layer_idx指定不同层的FC
        input_layer_fc_out = fluid.layers.fc(
            input=input_fc_out,
            size=self.node_emb_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="trans.layer_fc.weight." + str(layer_idx)),
            bias_attr=fluid.ParamAttr(
                name="trans.layer_fc.bias."+str(layer_idx)),
        )
        return input_layer_fc_out


class DnnLayerClassifierNet(object):
    """
    层次分类器的主要部分
    """

    def __init__(self, args):
        self.node_emb_size = args.node_emb_size
        self.max_layers = args.layer_size
        self.neg_sampling_list = args.neg_sampling_list
        self.output_positive = True
        self.is_test = args.is_test
        self.child_nums = args.child_nums

    def _expand_layer(self, input_layer, node, layer_idx):
        # 扩展input的输入，使数量与node一致，
        # 也可以以其他broadcast的操作进行代替
        # 同时兼容了训练组网与预测组网
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
        # 扩展input，使维度与node匹配
        input_expand = [
            self._expand_layer(input[i], node, i) for i in range(self.max_layers)
        ]

        # 将input_emb与node_emb concat到一起过分类器FC
        input_node_concat = [
            fluid.layers.concat(
                input=[input_expand[i], node[i]],
                axis=2) for i in range(self.max_layers)
        ]
        hidden_states_fc = [
            fluid.layers.fc(
                input=input_node_concat[i],
                size=self.node_emb_size,
                num_flatten_dims=2,
                act="tanh",
                param_attr=fluid.ParamAttr(
                    name="cls.concat_fc.weight."+str(i)),
                bias_attr=fluid.ParamAttr(name="cls.concat_fc.bias."+str(i))
            ) for i in range(self.max_layers)
        ]

        # 如果将所有层次的node放到一起计算loss，则需要在此处concat
        # 将分类器结果以batch为准绳concat到一起，而不是layer
        # 维度形如[batch_size, total_node_num, node_emb_size]
        hidden_states_concat = fluid.layers.concat(hidden_states_fc, axis=1)
        return hidden_states_concat

    def classifier_layer_infer(self, input, node, layer_idx):
        # 为infer组网提供的简化版classifier，通过给定layer_idx调用不同层的分类器

        # 同样需要保持input与node的维度匹配
        input_expand = self._expand_layer(input, node, layer_idx)

        # 与训练网络相同的concat逻辑
        input_node_concat = fluid.layers.concat(
            input=[input_expand, node], axis=2)

        # 根据参数名param_attr调用不同的层的FC
        hidden_states_fc = fluid.layers.fc(
            input=input_node_concat,
            size=self.node_emb_size,
            num_flatten_dims=2,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="cls.concat_fc.weight."+str(layer_idx)),
            bias_attr=fluid.ParamAttr(name="cls.concat_fc.bias."+str(layer_idx)))
        return hidden_states_fc
