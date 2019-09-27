#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from model.transformer_encoder import pre_process_layer
from utils.configure import JsonConfig


def compute_loss(output_tensors, args=None):
    """Compute loss for mlm model"""
    fc_out = output_tensors['mlm_out']
    mask_label = output_tensors['mask_label']
    mask_lm_loss = fluid.layers.softmax_with_cross_entropy(
        logits=fc_out, label=mask_label)
    mean_mask_lm_loss = fluid.layers.mean(mask_lm_loss)
    return mean_mask_lm_loss


def create_model(reader_input, base_model=None, is_training=True, args=None):
    """
        given the base model, reader_input
        return the output tensors
    """
    mask_label, mask_pos = reader_input

    config = JsonConfig(args.bert_config_path)

    _emb_size = config['hidden_size']
    _voc_size = config['vocab_size']
    _hidden_act = config['hidden_act']

    _word_emb_name = "word_embedding"
    _dtype = "float16" if args.use_fp16 else "float32"

    _param_initializer = fluid.initializer.TruncatedNormal(
        scale=config['initializer_range'])

    mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')

    enc_out = base_model.get_output("sequence_output")

    # extract the first token feature in each sentence
    reshaped_emb_out = fluid.layers.reshape(
        x=enc_out, shape=[-1, _emb_size])
    # extract masked tokens' feature
    mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)
    num_seqs = fluid.layers.fill_constant(shape=[1], value=512, dtype='int64')

    # transform: fc
    mask_trans_feat = fluid.layers.fc(
        input=mask_feat,
        size=_emb_size,
        act=_hidden_act,
        param_attr=fluid.ParamAttr(
            name='mask_lm_trans_fc.w_0',
            initializer=_param_initializer),
        bias_attr=fluid.ParamAttr(name='mask_lm_trans_fc.b_0'))
    # transform: layer norm
    mask_trans_feat = pre_process_layer(
        mask_trans_feat, 'n', name='mask_lm_trans')

    mask_lm_out_bias_attr = fluid.ParamAttr(
        name="mask_lm_out_fc.b_0",
        initializer=fluid.initializer.Constant(value=0.0))

    fc_out = fluid.layers.matmul(
        x=mask_trans_feat,
        y=fluid.default_main_program().global_block().var(
            _word_emb_name),
        transpose_y=True)
    fc_out += fluid.layers.create_parameter(
        shape=[_voc_size],
        dtype=_dtype,
        attr=mask_lm_out_bias_attr,
        is_bias=True)

    output_tensors = {}
    output_tensors['num_seqs'] = num_seqs
    output_tensors['mlm_out'] = fc_out
    output_tensors['mask_label'] = mask_label

    return output_tensors

