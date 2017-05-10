#!/usr/bin/env python
# -*- encoding:utf-8 -*-
# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
import math
import paddle.v2 as paddle


def network_conf(hidden_size, embed_size, dict_size, is_train):
    def wordemb(inlayer):
        wordemb = paddle.layer.table_projection(
            input=inlayer,
            size=embed_size,
            param_attr=paddle.attr.Param(
                name="_proj",
                initial_std=0.001,
                learning_rate=1,
                l2_rate=0,
                sparse_update=True))
        return wordemb

    firstword = paddle.layer.data(
        name="firstw", type=paddle.data_type.integer_value(dict_size))
    secondword = paddle.layer.data(
        name="secondw", type=paddle.data_type.integer_value(dict_size))
    thirdword = paddle.layer.data(
        name="thirdw", type=paddle.data_type.integer_value(dict_size))
    fourthword = paddle.layer.data(
        name="fourthw", type=paddle.data_type.integer_value(dict_size))
    nextword = paddle.layer.data(
        name="fifthw", type=paddle.data_type.integer_value(dict_size))

    Efirst = wordemb(firstword)
    Esecond = wordemb(secondword)
    Ethird = wordemb(thirdword)
    Efourth = wordemb(fourthword)

    contextemb = paddle.layer.concat(input=[Efirst, Esecond, Ethird, Efourth])

    hidden_layer = paddle.layer.fc(
        input=contextemb,
        size=hidden_size,
        act=paddle.activation.Sigmoid(),
        layer_attr=paddle.attr.Extra(drop_rate=0.5),
        bias_attr=paddle.attr.Param(learning_rate=1),
        param_attr=paddle.attr.Param(
            initial_std=1. / math.sqrt(embed_size * 8), learning_rate=1))

    predictword = paddle.layer.fc(
        input=hidden_layer,
        size=dict_size,
        bias_attr=paddle.attr.Param(learning_rate=1),
        act=paddle.activation.Softmax())

    if is_train == True:
        cost = paddle.layer.nce(
            name='softmax',
            input=predictword,
            label=nextword,
            num_classes=dict_size,
            num_neg_samples=10, )
        return cost
    else:
        with paddle.layer.mixed(
                size=dict_size,
                act=paddle.activation.Softmax(),
                bias_attr=paddle.attr.Param(
                    name='_softmax.wbias')) as prediction:
            prediction += paddle.layer.trans_full_matrix_projection(
                input=predictword,
                param_attr=paddle.attr.Param(name='_softmax.w0'))

        return prediction
