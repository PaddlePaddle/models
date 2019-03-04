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

import numpy as np
import paddle
import paddle.fluid as fluid
from . import clf_model


class NeXtVLAD(object):
    """
  This is a paddlepaddle implementation of the NeXtVLAD model. For more
  information, please refer to the paper,
   https://static.googleusercontent.com/media/research.google.com/zh-CN//youtube8m/workshop2018/p_c03.pdf
  """

    def __init__(self,
                 feature_size,
                 cluster_size,
                 is_training=True,
                 expansion=2,
                 groups=None,
                 inputname='video'):
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.is_training = is_training
        self.expansion = expansion
        self.groups = groups
        self.name = inputname + '_'

    def forward(self, input):
        input = fluid.layers.fc(
            input=input,
            size=self.expansion * self.feature_size,
            act=None,
            name=self.name + 'fc_expansion',
            param_attr=fluid.ParamAttr(
                name=self.name + 'fc_expansion_w',
                initializer=fluid.initializer.MSRA(uniform=False)),
            bias_attr=fluid.ParamAttr(
                name=self.name + 'fc_expansion_b',
                initializer=fluid.initializer.Constant(value=0.)))

        # attention factor of per group
        attention = fluid.layers.fc(
            input=input,
            size=self.groups,
            act='sigmoid',
            name=self.name + 'fc_group_attention',
            param_attr=fluid.ParamAttr(
                name=self.name + 'fc_group_attention_w',
                initializer=fluid.initializer.MSRA(uniform=False)),
            bias_attr=fluid.ParamAttr(
                name=self.name + 'fc_group_attention_b',
                initializer=fluid.initializer.Constant(value=0.)))

        # calculate activation factor of per group per cluster
        feature_size = self.feature_size * self.expansion // self.groups
        cluster_weights = fluid.layers.create_parameter(
            shape=[
                self.expansion * self.feature_size,
                self.groups * self.cluster_size
            ],
            dtype=input.dtype,
            attr=fluid.ParamAttr(name=self.name + 'cluster_weights'),
            default_initializer=fluid.initializer.MSRA(uniform=False))

        activation = fluid.layers.matmul(input, cluster_weights)
        activation = fluid.layers.batch_norm(
            activation, is_test=(not self.is_training))

        # reshape of activation
        activation = fluid.layers.reshape(activation,
                                          [-1, self.groups, self.cluster_size])
        # softmax on per cluster
        activation = fluid.layers.softmax(activation)
        activation = fluid.layers.elementwise_mul(activation, attention, axis=0)
        a_sum = fluid.layers.sequence_pool(activation, 'sum')
        a_sum = fluid.layers.reduce_sum(a_sum, dim=1)

        # create cluster_weights2
        cluster_weights2 = fluid.layers.create_parameter(
            shape=[self.cluster_size, feature_size],
            dtype=input.dtype,
            attr=fluid.ParamAttr(name=self.name + 'cluster_weights2'),
            default_initializer=fluid.initializer.MSRA(uniform=False))

        # expand a_sum dimension from [-1, self.cluster_size] to be [-1, self.cluster_size, feature_size]
        a_sum = fluid.layers.reshape(a_sum, [-1, self.cluster_size, 1])
        a_sum = fluid.layers.expand(a_sum, [1, 1, feature_size])

        # element wise multiply a_sum and cluster_weights2
        a = fluid.layers.elementwise_mul(
            a_sum, cluster_weights2,
            axis=1)  # output shape [-1, self.cluster_size, feature_size]

        # transpose activation from [-1, self.groups, self.cluster_size] to [-1, self.cluster_size, self.groups]
        activation2 = fluid.layers.transpose(activation, perm=[0, 2, 1])
        # transpose op will clear the lod infomation, so it should be reset
        activation = fluid.layers.lod_reset(activation2, activation)

        # reshape input from [-1, self.expansion * self.feature_size] to [-1, self.groups, feature_size]
        reshaped_input = fluid.layers.reshape(input,
                                              [-1, self.groups, feature_size])
        # mat multiply activation and reshaped_input
        vlad = fluid.layers.matmul(
            activation,
            reshaped_input)  # output shape [-1, self.cluster_size, feature_size]
        vlad = fluid.layers.sequence_pool(vlad, 'sum')
        vlad = fluid.layers.elementwise_sub(vlad, a)

        # l2_normalization
        vlad = fluid.layers.transpose(vlad, [0, 2, 1])
        vlad = fluid.layers.l2_normalize(vlad, axis=1)

        # reshape and batch norm
        vlad = fluid.layers.reshape(vlad,
                                    [-1, self.cluster_size * feature_size])
        vlad = fluid.layers.batch_norm(vlad, is_test=(not self.is_training))

        return vlad


class NeXtVLADModel(object):
    """
  Creates a NeXtVLAD based model.
  Args:
    model_input: A LoDTensor of [-1, N] for the input video frames.
    vocab_size: The number of classes in the dataset.
  """

    def __init__(self):
        pass

    def create_model(self,
                     video_input,
                     audio_input,
                     is_training=True,
                     class_dim=None,
                     cluster_size=None,
                     hidden_size=None,
                     groups=None,
                     expansion=None,
                     drop_rate=None,
                     gating_reduction=None,
                     l2_penalty=None,
                     **unused_params):

        # calcluate vlad of video and audio
        video_nextvlad = NeXtVLAD(
            1024,
            cluster_size,
            is_training,
            expansion=expansion,
            groups=groups,
            inputname='video')
        audio_nextvlad = NeXtVLAD(
            128,
            cluster_size,
            is_training,
            expansion=expansion,
            groups=groups,
            inputname='audio')
        vlad_video = video_nextvlad.forward(video_input)
        vlad_audio = audio_nextvlad.forward(audio_input)

        # concat video and audio
        vlad = fluid.layers.concat([vlad_video, vlad_audio], axis=1)

        # drop out
        if drop_rate > 0.:
            vlad = fluid.layers.dropout(
                vlad, drop_rate, is_test=(not is_training))

        # add fc
        activation = fluid.layers.fc(
            input=vlad,
            size=hidden_size,
            act=None,
            name='hidden1_fc',
            param_attr=fluid.ParamAttr(
                name='hidden1_fc_weights',
                initializer=fluid.initializer.MSRA(uniform=False)),
            bias_attr=False)
        activation = fluid.layers.batch_norm(
            activation, is_test=(not is_training))

        # add fc, gate 1
        gates = fluid.layers.fc(
            input=activation,
            size=hidden_size // gating_reduction,
            act=None,
            name='gating_fc1',
            param_attr=fluid.ParamAttr(
                name='gating_fc1_weights',
                initializer=fluid.initializer.MSRA(uniform=False)),
            bias_attr=False)
        gates = fluid.layers.batch_norm(
            gates, is_test=(not is_training), act='relu')

        # add fc, gate 2
        gates = fluid.layers.fc(
            input=gates,
            size=hidden_size,
            act='sigmoid',
            name='gating_fc2',
            param_attr=fluid.ParamAttr(
                name='gating_fc2_weights',
                initializer=fluid.initializer.MSRA(uniform=False)),
            bias_attr=False)

        activation = fluid.layers.elementwise_mul(activation, gates)
        aggregate_model = clf_model.LogisticModel  # set classification model

        return aggregate_model().create_model(
            model_input=activation,
            vocab_size=class_dim,
            l2_penalty=l2_penalty,
            is_training=is_training,
            **unused_params)
