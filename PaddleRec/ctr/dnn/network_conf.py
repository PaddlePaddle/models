#!/usr/bin/python
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

import paddle.fluid as fluid
import math


class CTR(object):
    """
    DNN for Click-Through Rate prediction
    """

    def input_data(self, args):
        dense_input = fluid.data(name="dense_input",
                                 shape=[-1, args.dense_feature_dim],
                                 dtype="float32")

        sparse_input_ids = [
            fluid.data(name="C" + str(i),
                       shape=[-1, 1],
                       lod_level=1,
                       dtype="int64") for i in range(1, 27)
        ]

        label = fluid.data(name="label", shape=[-1, 1], dtype="int64")

        inputs = [dense_input] + sparse_input_ids + [label]
        return inputs

    def net(self, inputs, args):
        def embedding_layer(input):
            return fluid.layers.embedding(
                input=input,
                is_sparse=True,
                size=[args.sparse_feature_dim, args.embedding_size],
                param_attr=fluid.ParamAttr(
                    name="SparseFeatFactors",
                    initializer=fluid.initializer.Uniform()),
            )

        sparse_embed_seq = list(map(embedding_layer, inputs[1:-1]))

        concated = fluid.layers.concat(sparse_embed_seq + inputs[0:1], axis=1)

        fc1 = fluid.layers.fc(
            input=concated,
            size=400,
            act="relu",
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(concated.shape[1]))),
        )
        fc2 = fluid.layers.fc(
            input=fc1,
            size=400,
            act="relu",
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(fc1.shape[1]))),
        )
        fc3 = fluid.layers.fc(
            input=fc2,
            size=400,
            act="relu",
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(fc2.shape[1]))),
        )
        predict = fluid.layers.fc(
            input=fc3,
            size=2,
            act="softmax",
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(fc3.shape[1]))),
        )

        cost = fluid.layers.cross_entropy(input=predict, label=inputs[-1])
        avg_cost = fluid.layers.reduce_sum(cost)
        auc_var, _, _ = fluid.layers.auc(input=predict,
                                         label=inputs[-1],
                                         num_thresholds=2**12,
                                         slide_steps=20)

        return avg_cost, auc_var
