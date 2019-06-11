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
dense_feature_dim = 3

def ctr_deepfm_dataset(dense_feature, context_feature, context_feature_fm, label,
                       embedding_size, sparse_feature_dim):
    def dense_fm_layer(input, emb_dict_size, factor_size, fm_param_attr):

        first_order = fluid.layers.fc(input=input, size=1)
        emb_table = fluid.layers.create_parameter(shape=[emb_dict_size, factor_size],
                                                  dtype='float32', attr=fm_param_attr)

        input_mul_factor = fluid.layers.matmul(input, emb_table)
        input_mul_factor_square = fluid.layers.square(input_mul_factor)
        input_square = fluid.layers.square(input)
        factor_square = fluid.layers.square(emb_table)
        input_square_mul_factor_square = fluid.layers.matmul(input_square, factor_square)

        second_order = 0.5 * (input_mul_factor_square - input_square_mul_factor_square)
        return first_order, second_order


    dense_fm_param_attr = fluid.param_attr.ParamAttr(name="DenseFeatFactors",
                                                     initializer=fluid.initializer.Normal(
                                                         scale=1 / math.sqrt(dense_feature_dim)))
    dense_fm_first, dense_fm_second = dense_fm_layer(
        dense_feature, dense_feature_dim, 16, dense_fm_param_attr)


    def sparse_fm_layer(input, emb_dict_size, factor_size, fm_param_attr):

        first_embeddings = fluid.layers.embedding(
            input=input, dtype='float32', size=[emb_dict_size, 1], is_sparse=True)
        first_order = fluid.layers.sequence_pool(input=first_embeddings, pool_type='sum')

        nonzero_embeddings = fluid.layers.embedding(
            input=input, dtype='float32', size=[emb_dict_size, factor_size],
            param_attr=fm_param_attr, is_sparse=True)
        summed_features_emb = fluid.layers.sequence_pool(input=nonzero_embeddings, pool_type='sum')
        summed_features_emb_square = fluid.layers.square(summed_features_emb)

        squared_features_emb = fluid.layers.square(nonzero_embeddings)
        squared_sum_features_emb = fluid.layers.sequence_pool(
            input=squared_features_emb, pool_type='sum')

        second_order = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
        return first_order, second_order

    sparse_fm_param_attr = fluid.param_attr.ParamAttr(name="SparseFeatFactors",
                                                      initializer=fluid.initializer.Normal(
                                                          scale=1 / math.sqrt(sparse_feature_dim)))

    #data = fluid.layers.data(name='ids', shape=[1], dtype='float32')
    sparse_fm_first, sparse_fm_second = sparse_fm_layer(
        context_feature_fm, sparse_feature_dim, 16, sparse_fm_param_attr)

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

    concated_ori = fluid.layers.concat(sparse_embed_seq + [dense_feature], axis=1)
    concated = fluid.layers.batch_norm(input=concated_ori, name="bn", epsilon=1e-4)

    deep = deep_net(concated)

    predict = fluid.layers.fc(input=[deep, sparse_fm_first, sparse_fm_second, dense_fm_first, dense_fm_second], size=2, act="softmax",
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
    fc_layers_size = [400, 400, 400]
    fc_layers_act = ["relu"] * (len(fc_layers_size))

    for i in range(len(fc_layers_size)):
        fc = fluid.layers.fc(
            input=fc_layers_input[-1],
            size=fc_layers_size[i],
            act=fc_layers_act[i],
            param_attr=fluid.ParamAttr(learning_rate=lr_x * 0.5))

        fc_layers_input.append(fc)
    #w_res = fluid.layers.create_parameter(shape=[353, 16], dtype='float32', name="w_res")
    #high_path = fluid.layers.matmul(concated, w_res)

    #return fluid.layers.elementwise_add(high_path, fc_layers_input[-1])
    return fc_layers_input[-1]