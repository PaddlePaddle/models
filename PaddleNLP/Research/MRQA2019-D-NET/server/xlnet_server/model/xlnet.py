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
"""BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import json
import numpy as np
import paddle.fluid as fluid
from model.transformer_encoder import encoder, pre_process_layer
import modeling

def _get_initiliaizer(args):
    if args.init == "uniform":
        param_initializer = fluid.initializer.Uniform(
            low=-args.init_range, high=args.init_range)
    elif args.init == "normal":
        param_initializer = fluid.initializer.Normal(scale=args.init_std)
    else:
        raise ValueError("Initializer {} not supported".format(args.init))
    return param_initializer
      
def init_attn_mask(args, place):
    """create causal attention mask."""
    qlen = args.max_seq_length
    mlen=0 if 'mem_len' not in args else args.mem_len
    same_length=False if 'same_length' not in args else args.same_length
    dtype = 'float16' if args.use_fp16 else 'float32'
    attn_mask = np.ones([qlen, qlen], dtype=dtype)
    mask_u = np.triu(attn_mask)
    mask_dia = np.diag(np.diag(attn_mask))
    attn_mask_pad = np.zeros([qlen, mlen], dtype=dtype)
    attn_mask = np.concatenate([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
        mask_l = np.tril(attn_mask)
        attn_mask = np.concatenate([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
    attn_mask = attn_mask[:, :, None, None]
    attn_mask_t = fluid.global_scope().find_var("attn_mask").get_tensor()
    attn_mask_t.set(attn_mask, place)

class XLNetConfig(object):
    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing xlnet model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict[key]

    def has_key(self, key):
        return self._config_dict.has_key(key)

    def print_config(self):
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class XLNetModel(object):
    def __init__(self,
                 xlnet_config,
                 input_ids,
                 seg_ids,
                 input_mask,
                 args,
                 mems=None,
                 perm_mask=None,
                 target_mapping=None,
                 inp_q=None):
        self._tie_weight = True

        self._d_head = xlnet_config['d_head']
        self._d_inner = xlnet_config['d_inner']
        self._d_model = xlnet_config['d_model']
        self._ff_activation = xlnet_config['ff_activation']
        self._n_head = xlnet_config['n_head']
        self._n_layer = xlnet_config['n_layer']
        self._n_token = xlnet_config['n_token']
        self._untie_r = xlnet_config['untie_r']

        self._mem_len=None if 'mem_len' not in args else args.mem_len
        self._reuse_len=None if 'reuse_len' not in args else args.reuse_len
        self._bi_data=False if 'bi_data' not in args else args.bi_data
        self._clamp_len=args.clamp_len
        self._same_length=False if 'same_length' not in args else args.same_length
        # Initialize all weigths by the specified initializer, and all biases 
        # will be initialized by constant zero by default.
        self._param_initializer = _get_initiliaizer(args)

        tfm_args = dict(
                n_token=self._n_token,
                initializer=self._param_initializer,
                attn_type="bi",
                n_layer=self._n_layer,
                d_model=self._d_model,
		n_head=self._n_head,
		d_head=self._d_head,
		d_inner=self._d_inner,
		ff_activation=self._ff_activation,
		untie_r=self._untie_r,

		use_bfloat16=args.use_fp16,
		dropout=args.dropout,
		dropatt=args.dropatt,

		mem_len=self._mem_len,
		reuse_len=self._reuse_len,
		bi_data=self._bi_data,
		clamp_len=args.clamp_len,
		same_length=self._same_length,
                name='model_transformer')
        input_args = dict(
            inp_k=input_ids,
            seg_id=seg_ids,
            input_mask=input_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            inp_q=inp_q)
        tfm_args.update(input_args)
        self.output, self.new_mems, self.lookup_table = modeling.transformer_xl(**tfm_args)
        #self._build_model(input_ids, sentence_ids, input_mask)

    def get_initializer(self):
        return self._param_initializer

    
     
    def get_sequence_output(self):
        return self.output

    def get_pooled_output(self):
        """Get the first feature of each sequence for classification"""

        next_sent_feat = fluid.layers.slice(
            input=self._enc_out, axes=[1], starts=[0], ends=[1])
        next_sent_feat = fluid.layers.fc(
            input=next_sent_feat,
            size=self._emb_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="pooled_fc.w_0", initializer=self._param_initializer),
            bias_attr="pooled_fc.b_0")
        return next_sent_feat

    def get_pretraining_output(self, mask_label, mask_pos, labels):
        """Get the loss & accuracy for pretraining"""

        mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')

        # extract the first token feature in each sentence
        next_sent_feat = self.get_pooled_output()
        reshaped_emb_out = fluid.layers.reshape(
            x=self._enc_out, shape=[-1, self._emb_size])
        # extract masked tokens' feature
        mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)

        # transform: fc
        mask_trans_feat = fluid.layers.fc(
            input=mask_feat,
            size=self._emb_size,
            act=self._hidden_act,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_fc.w_0',
                initializer=self._param_initializer),
            bias_attr=fluid.ParamAttr(name='mask_lm_trans_fc.b_0'))
        # transform: layer norm 
        mask_trans_feat = pre_process_layer(
            mask_trans_feat, 'n', name='mask_lm_trans')

        mask_lm_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))
        if self._weight_sharing:
            word_emb = fluid.default_main_program().global_block().var(
                self._word_emb_name)
            if self._emb_dtype != self._dtype:
                word_emb = fluid.layers.cast(word_emb, self._dtype)
            fc_out = fluid.layers.matmul(
                x=mask_trans_feat, y=word_emb, transpose_y=True)
            fc_out += fluid.layers.create_parameter(
                shape=[self._voc_size],
                dtype=self._dtype,
                attr=mask_lm_out_bias_attr,
                is_bias=True)

        else:
            fc_out = fluid.layers.fc(input=mask_trans_feat,
                                     size=self._voc_size,
                                     param_attr=fluid.ParamAttr(
                                         name="mask_lm_out_fc.w_0",
                                         initializer=self._param_initializer),
                                     bias_attr=mask_lm_out_bias_attr)

        mask_lm_loss = fluid.layers.softmax_with_cross_entropy(
            logits=fc_out, label=mask_label)
        mean_mask_lm_loss = fluid.layers.mean(mask_lm_loss)

        next_sent_fc_out = fluid.layers.fc(
            input=next_sent_feat,
            size=2,
            param_attr=fluid.ParamAttr(
                name="next_sent_fc.w_0", initializer=self._param_initializer),
            bias_attr="next_sent_fc.b_0")

        next_sent_loss, next_sent_softmax = fluid.layers.softmax_with_cross_entropy(
            logits=next_sent_fc_out, label=labels, return_softmax=True)

        next_sent_acc = fluid.layers.accuracy(
            input=next_sent_softmax, label=labels)

        mean_next_sent_loss = fluid.layers.mean(next_sent_loss)

        loss = mean_next_sent_loss + mean_mask_lm_loss
        return next_sent_acc, mean_mask_lm_loss, loss
