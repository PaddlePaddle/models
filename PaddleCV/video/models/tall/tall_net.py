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
import numpy as np


class TALLNET(object):
    def __init__(self,
                 semantic_size,
                 sentence_embedding_size,
                 hidden_size,
                 output_size,
                 batch_size,
                 mode='train'):

        self.semantic_size = semantic_size
        self.sentence_embedding_size = sentence_embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size  #divide train and test

        self.mode = mode

    def cross_modal_comb(self, visual_feat, sentence_embed):
        visual_feat = fluid.layers.reshape(visual_feat,
                                           [1, -1, self.semantic_size])
        vv_feature = fluid.layers.expand(visual_feat, [self.batch_size, 1, 1])
        sentence_embed = fluid.layers.reshape(sentence_embed,
                                              [-1, 1, self.semantic_size])
        ss_feature = fluid.layers.expand(sentence_embed,
                                         [1, self.batch_size, 1])

        concat_feature = fluid.layers.concat(
            [vv_feature, ss_feature], axis=2)  #B,B,2048

        mul_feature = vv_feature * ss_feature  # B,B,1024
        add_feature = vv_feature + ss_feature  # B,B,1024

        comb_feature = fluid.layers.concat(
            [mul_feature, add_feature, concat_feature], axis=2)
        return comb_feature

    def net(self, images, sentences):
        # visual2semantic
        transformed_clip = fluid.layers.fc(
            input=images,
            size=self.semantic_size,
            act=None,
            name='v2s_lt',
            param_attr=fluid.ParamAttr(
                name='v2s_lt_weights',
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=1.0, seed=0)),
            bias_attr=False)

        #l2_normalize
        transformed_clip = fluid.layers.l2_normalize(x=transformed_clip, axis=1)

        # sentenct2semantic
        transformed_sentence = fluid.layers.fc(
            input=sentences,
            size=self.semantic_size,
            act=None,
            name='s2s_lt',
            param_attr=fluid.ParamAttr(
                name='s2s_lt_weights',
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=1.0, seed=0)),
            bias_attr=False)

        #l2_normalize
        transformed_sentence = fluid.layers.l2_normalize(
            x=transformed_sentence, axis=1)

        cross_modal_vec = self.cross_modal_comb(transformed_clip,
                                                transformed_sentence)
        cross_modal_vec = fluid.layers.unsqueeze(
            input=cross_modal_vec, axes=[0])
        cross_modal_vec = fluid.layers.transpose(
            cross_modal_vec, perm=[0, 3, 1, 2])

        mid_output = fluid.layers.conv2d(
            input=cross_modal_vec,
            num_filters=self.hidden_size,
            filter_size=1,
            stride=1,
            act="relu",
            param_attr=fluid.param_attr.ParamAttr(name="mid_out_weights"),
            bias_attr=False)

        sim_score_mat = fluid.layers.conv2d(
            input=mid_output,
            num_filters=self.output_size,
            filter_size=1,
            stride=1,
            act=None,
            param_attr=fluid.param_attr.ParamAttr(name="sim_mat_weights"),
            bias_attr=False)
        sim_score_mat = fluid.layers.squeeze(input=sim_score_mat, axes=[0])
        return sim_score_mat

    def loss(self, outs, offs):
        sim_score_mat = outs[0]
        p_reg_mat = outs[1]
        l_reg_mat = outs[2]
        # loss cls, not considering iou
        input_size = outs.shape[1]
        I = fluid.layers.diag(np.array([1] * input_size).astype('float32'))
        I_2 = -2 * I
        all1 = fluid.layers.ones(
            shape=[input_size, input_size], dtype="float32")

        mask_mat = I_2 + all1

        alpha = 1.0 / input_size
        lambda_regression = 0.01
        batch_para_mat = alpha * all1
        para_mat = I + batch_para_mat

        sim_mask_mat = fluid.layers.exp(mask_mat * sim_score_mat)
        loss_mat = fluid.layers.log(all1 + sim_mask_mat)
        loss_mat = loss_mat * para_mat
        loss_align = fluid.layers.mean(loss_mat)

        # regression loss
        reg_ones = fluid.layers.ones(shape=[input_size, 1], dtype="float32")
        l_reg_diag = fluid.layers.matmul(
            l_reg_mat * I, reg_ones, transpose_x=True, transpose_y=False)
        p_reg_diag = fluid.layers.matmul(
            p_reg_mat * I, reg_ones, transpose_x=True, transpose_y=False)
        offset_pred = fluid.layers.concat(
            input=[p_reg_diag, l_reg_diag], axis=1)
        loss_reg = fluid.layers.mean(
            fluid.layers.abs(offset_pred - offs))  # L1 loss
        loss = lambda_regression * loss_reg + loss_align
        avg_loss = fluid.layers.mean(loss)

        return [avg_loss]
