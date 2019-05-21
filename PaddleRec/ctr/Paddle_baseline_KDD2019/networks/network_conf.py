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


import paddle.fluid as fluid
import math

user_profile_dim = 65
num_context = 25
dim_fm_vector = 16
dim_concated = user_profile_dim + dim_fm_vector * (num_context)


def ctr_deepfm_dataset(user_profile, context_feature, label,
                       embedding_size, sparse_feature_dim):
    def embedding_layer(input):
        return fluid.layers.embedding(
            input=input,
            is_sparse=True,
            # you need to patch https://github.com/PaddlePaddle/Paddle/pull/14190
            # if you want to set is_distributed to True
            is_distributed=False,
            size=[sparse_feature_dim, embedding_size],
            param_attr=fluid.ParamAttr(name="SparseFeatFactors",
                                       initializer=fluid.initializer.Uniform()))

    sparse_embed_seq = list(map(embedding_layer, context_feature))

    w = fluid.layers.create_parameter(
        shape=[65, 65], dtype='float32',
        name="w_fm")
    user_profile_emb = fluid.layers.matmul(user_profile, w)

    concated_ori = fluid.layers.concat(sparse_embed_seq + [user_profile_emb], axis=1)
    concated = fluid.layers.batch_norm(input=concated_ori, name="bn", epsilon=1e-4)

    deep = deep_net(concated)
    linear_term, second_term = fm(concated, dim_concated, 8) #depend on the number of context feature

    predict = fluid.layers.fc(input=[deep, linear_term, second_term], size=2, act="softmax",
                              param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                  scale=1 / math.sqrt(deep.shape[1])), learning_rate=0.01))

    #similarity_norm = fluid.layers.sigmoid(fluid.layers.clip(predict, min=-15.0, max=15.0), name="similarity_norm")


    cost = fluid.layers.cross_entropy(input=predict, label=label)

    avg_cost = fluid.layers.reduce_sum(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    auc_var, batch_auc_var, auc_states = \
        fluid.layers.auc(input=predict, label=label, num_thresholds=2 ** 12, slide_steps=20)
    return avg_cost, auc_var, batch_auc_var, accuracy, predict


def deep_net(concated, lr_x=0.0001):
    fc_layers_input = [concated]
    fc_layers_size = [128, 64, 32, 16]
    fc_layers_act = ["relu"] * (len(fc_layers_size))

    for i in range(len(fc_layers_size)):
        fc = fluid.layers.fc(
            input=fc_layers_input[-1],
            size=fc_layers_size[i],
            act=fc_layers_act[i],
            param_attr=fluid.ParamAttr(learning_rate=lr_x * 0.5))

        fc_layers_input.append(fc)

    return fc_layers_input[-1]


def fm(concated, emb_dict_size, factor_size, lr_x=0.0001):
    linear_term = fluid.layers.fc(input=concated, size=8, act=None, param_attr=fluid.ParamAttr(learning_rate=lr_x))

    emb_table = fluid.layers.create_parameter(shape=[emb_dict_size, factor_size],
                                                  dtype='float32')

    input_mul_factor = fluid.layers.matmul(concated, emb_table)
    input_mul_factor_square = fluid.layers.square(input_mul_factor)
    input_square = fluid.layers.square(concated)
    factor_square = fluid.layers.square(emb_table)
    input_square_mul_factor_square = fluid.layers.matmul(input_square, factor_square)

    second_term = 0.5 * (input_mul_factor_square - input_square_mul_factor_square)

    return linear_term, second_term








