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

import paddle.fluid as fluid
from paddle.fluid import ParamAttr

from ..model import ModelBase
from .shifting_attention import ShiftingAttentionModel
from .logistic_model import LogisticModel

__all__ = ["AttentionCluster"]


class AttentionCluster(ModelBase):
    def __init__(self, name, cfg, mode='train'):
        super(AttentionCluster, self).__init__(name, cfg, mode)
        self.get_config()

    def get_config(self):
        # get model configs
        self.feature_num = self.cfg.MODEL.feature_num
        self.feature_names = self.cfg.MODEL.feature_names
        self.feature_dims = self.cfg.MODEL.feature_dims
        self.cluster_nums = self.cfg.MODEL.cluster_nums
        self.seg_num = self.cfg.MODEL.seg_num
        self.class_num = self.cfg.MODEL.num_classes
        self.drop_rate = self.cfg.MODEL.drop_rate

        if self.mode == 'train':
            self.learning_rate = self.get_config_from_sec('train',
                                                          'learning_rate', 1e-3)

    def build_input(self, use_dataloader=True):
        self.feature_input = []
        for name, dim in zip(self.feature_names, self.feature_dims):
            self.feature_input.append(
                fluid.data(
                    shape=[None, self.seg_num, dim], dtype='float32',
                    name=name))
        if self.mode != 'infer':
            self.label_input = fluid.data(
                shape=[None, self.class_num], dtype='float32', name='label')
        else:
            self.label_input = None
        if use_dataloader:
            assert self.mode != 'infer', \
                    'dataloader is not recommendated when infer, please set use_dataloader to be false.'
            self.dataloader = fluid.io.DataLoader.from_generator(
                feed_list=self.feature_input + [self.label_input],
                capacity=8,
                iterable=True)

    def build_model(self):
        att_outs = []
        for i, (input_dim, cluster_num, feature) in enumerate(
                zip(self.feature_dims, self.cluster_nums, self.feature_input)):
            att = ShiftingAttentionModel(input_dim, self.seg_num, cluster_num,
                                         "satt{}".format(i))
            att_out = att.forward(feature)
            att_outs.append(att_out)
        out = fluid.layers.concat(att_outs, axis=1)

        if self.drop_rate > 0.:
            out = fluid.layers.dropout(
                out, self.drop_rate, is_test=(not self.is_training))

        fc1 = fluid.layers.fc(
            out,
            size=1024,
            act='tanh',
            param_attr=ParamAttr(
                name="fc1.weights",
                initializer=fluid.initializer.MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name="fc1.bias", initializer=fluid.initializer.MSRA()))
        fc2 = fluid.layers.fc(
            fc1,
            size=4096,
            act='tanh',
            param_attr=ParamAttr(
                name="fc2.weights",
                initializer=fluid.initializer.MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name="fc2.bias", initializer=fluid.initializer.MSRA()))

        aggregate_model = LogisticModel()

        self.output, self.logit = aggregate_model.build_model(
            model_input=fc2,
            vocab_size=self.class_num,
            is_training=self.is_training)

    def optimizer(self):
        assert self.mode == 'train', "optimizer only can be get in train mode"
        return fluid.optimizer.AdamOptimizer(self.learning_rate)

    def loss(self):
        assert self.mode != 'infer', "invalid loss calculationg in infer mode"
        cost = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=self.logit, label=self.label_input)
        cost = fluid.layers.reduce_sum(cost, dim=-1)
        self.loss_ = fluid.layers.mean(x=cost)
        return self.loss_

    def outputs(self):
        return [self.output, self.logit]

    def feeds(self):
        return self.feature_input if self.mode == 'infer' else self.feature_input + [
            self.label_input
        ]

    def fetches(self):
        if self.mode == 'train' or self.mode == 'valid':
            losses = self.loss()
            fetch_list = [losses, self.output, self.label_input]
        elif self.mode == 'test':
            losses = self.loss()
            fetch_list = [losses, self.output, self.label_input]
        elif self.mode == 'infer':
            fetch_list = [self.output]
        else:
            raise NotImplementedError('mode {} not implemented'.format(
                self.mode))

        return fetch_list

    def weights_info(self):
        return (
            "AttentionCluster.pdparams",
            "https://paddlemodels.bj.bcebos.com/video_classification/AttentionCluster.pdparams"
        )
