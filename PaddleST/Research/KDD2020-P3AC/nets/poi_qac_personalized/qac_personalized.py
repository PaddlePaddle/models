#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
################################################################################


"""
 Specify the brief poi_qac_personalized.py
"""

import math
import numpy as np
import logging
import collections
import paddle.fluid as fluid

from nets.base_net import BaseNet


def ffn(input, d_hid, d_size, name="ffn"):
    """
        Position-wise Feed-Forward Network
    """
    hidden = fluid.layers.fc(input=input,
             size=d_hid,
             num_flatten_dims=1,
             param_attr=fluid.ParamAttr(name=name + '_innerfc_weight'),
             bias_attr=fluid.ParamAttr(
                 name=name + '_innerfc_bias',
                 initializer=fluid.initializer.Constant(0.)),
             act="leaky_relu")

    out = fluid.layers.fc(input=hidden,
              size=d_size,
              num_flatten_dims=1,
              param_attr=fluid.ParamAttr(name=name + '_outerfc_weight'),
              bias_attr=fluid.ParamAttr(
                  name=name + '_outerfc_bias',
                  initializer=fluid.initializer.Constant(0.)))
    return out


def dot_product_attention(query, key, value, d_key, q_mask=None, k_mask=None,
        dropout_rate=None):
    """
     Args:
         query: a tensor with shape [batch, Q_time, Q_dimension]
         key: a tensor with shape [batch, time, K_dimension]
         value: a tensor with shape [batch, time, V_dimension]

         q_lengths: a tensor with shape [batch]
         k_lengths: a tensor with shape [batch]

     Returns:
         a tensor with shape [batch, query_time, value_dimension]

     Raises:
         AssertionError: if Q_dimension not equal to K_dimension when attention 
                        type is dot.
    """

    logits = fluid.layers.matmul(x=query, y=key, transpose_y=True, alpha=d_key**(-0.5))

    if (q_mask is not None) and (k_mask is not None):
        mask = fluid.layers.matmul(x=q_mask, y=k_mask, transpose_y=True)
        another_mask = fluid.layers.scale(
            mask,
            scale=float(2**32 - 1),
            bias=float(-1),
            bias_after_scale=False)

        logits = mask * logits + another_mask

    attention = fluid.layers.softmax(logits)
    if dropout_rate:
        attention = fluid.layers.dropout(
            input=attention, dropout_prob=dropout_rate, is_test=False)

    atten_out = fluid.layers.matmul(x=attention, y=value)

    return atten_out


def safe_cosine_sim(x, y):
    """
        fluid.layers.cos_sim maybe nan
        avoid nan
    """
    l2x = fluid.layers.l2_normalize(x, axis=-1)
    l2y = fluid.layers.l2_normalize(y, axis=-1)
    cos = fluid.layers.reduce_sum(l2x * l2y, dim=1, keep_dim=True)
    return cos


def loss_neg_log_of_pos(pos_score, neg_score_n, gama=5.0):
    '''
        pos_score: batch_size x 1
        neg_score_n: batch_size x n
    '''
    # n x batch_size
    neg_score_n = fluid.layers.transpose(neg_score_n, [1, 0])
    # 1 x batch_size
    pos_score = fluid.layers.reshape(pos_score, [1, -1])

    exp_pos_score = fluid.layers.exp(pos_score * gama)
    exp_neg_score_n = fluid.layers.exp(neg_score_n * gama)

    ## (n+1) x batch_size
    pos_neg_score = fluid.layers.concat([exp_pos_score, exp_neg_score_n], axis=0)
    ## 1 x batch_size
    exp_sum = fluid.layers.reduce_sum(pos_neg_score, dim=0, keep_dim=True)
    ## 1 x batch_size
    loss = -1.0 * fluid.layers.log(exp_pos_score / exp_sum)
    # batch_size
    loss = fluid.layers.reshape(loss, [-1, 1])
    #return [loss, exp_pos_score, exp_neg_score_n, pos_neg_score, exp_sum]
    return loss


def loss_pairwise_hinge(pos, neg, margin=0.8):
    """
        pairwise
    """
    loss_part1 = fluid.layers.elementwise_sub(
        fluid.layers.fill_constant_batch_size_like(
            input=pos, shape=[-1, 1], value=margin, dtype='float32'), pos)
    loss_part2 = fluid.layers.elementwise_add(loss_part1, neg)
    loss_part3 = fluid.layers.elementwise_max(
        fluid.layers.fill_constant_batch_size_like(
            input=loss_part2, shape=[-1, 1], value=0.0, dtype='float32'), loss_part2)
    return loss_part3


class PoiQacPersonalized(BaseNet):
    """
    This module provide nets for poi classification
    """
    def __init__(self, FLAGS):
        super(PoiQacPersonalized, self).__init__(FLAGS)
        self.hid_dim = 128

    def net(self, inputs):
        """
        PoiQacPersonalized interface
        """ 
        # debug output info during training
        debug_output = collections.OrderedDict()
        model_output = {}
        net_output = {"debug_output": debug_output, 
                      "model_output": model_output}
       
        user_input_keys = ['user_clk_geoid', 'user_tag_id', 'user_resident_geoid', 'user_navi_drive']
            
        pred_input_keys = ['prefix_letter_id', 'prefix_loc_geoid', 'pos_name_letter_id',
            'pos_name_word_id', 'pos_addr_letter_id', 'pos_addr_word_id', 'pos_loc_geoid']
        query_key_num = 2
        if self._flags.use_personal:
            pred_input_keys.insert(2, 'prefix_time_id')
            pred_input_keys.append('pos_tag_id')
            query_key_num += 2
            if self._flags.prefix_word_id:
                pred_input_keys.insert(1, 'prefix_word_id')
                query_key_num += 1
            pred_input_keys = user_input_keys + pred_input_keys
            query_key_num += len(user_input_keys)
        elif self._flags.prefix_word_id:
            pred_input_keys.insert(1, 'prefix_word_id')
            query_key_num += 1
        
        #for p in pred_input_keys:
        #    debug_output[p] = inputs[p]

        prefix_vec, prefix_pool = self._get_query_vec(inputs)
        pos_vec, pos_pool = self._get_poi_vec(inputs, 'pos')
        
        pos_score = safe_cosine_sim(pos_vec, prefix_vec)
        #fluid.layers.Print(pos_score, summarize=10000)
        if self.is_training:
            neg_vec, neg_pool = self._get_poi_vec(inputs, 'neg') 
            if self._flags.loss_func == 'log_exp':
                neg_vec = fluid.layers.reshape(neg_vec, [-1, self._flags.fc_dim])
                prefix_expand = fluid.layers.reshape(fluid.layers.expand(prefix_vec, [1,
                                self._flags.neg_sample_num]), [-1, self._flags.fc_dim])
                neg_score = safe_cosine_sim(neg_vec, prefix_expand)
                cost = loss_neg_log_of_pos(pos_score,  fluid.layers.reshape(neg_score,
                            [-1, self._flags.neg_sample_num]), 15) 
            else:
                neg_score = safe_cosine_sim(neg_vec, prefix_vec)
                cost = loss_pairwise_hinge(pos_score, neg_score, self._flags.margin)
            #debug_output["pos_score"] = pos_score
            #debug_output["neg_score"] = neg_score
            #debug_output['prefix_pool'] = prefix_pool
            #debug_output['pos_pool'] = pos_pool
            #debug_output['neg_pool'] = neg_pool

            loss = fluid.layers.mean(x=cost)
            if self._flags.init_learning_rate > 0:
                # define the optimizer
                #d_model = 1 / (warmup_steps * (learning_rate ** 2)) 
                with fluid.default_main_program()._lr_schedule_guard():
                    learning_rate = fluid.layers.learning_rate_scheduler.noam_decay(
                        self._flags.emb_dim, self._flags.learning_rate_warmup_steps
                        ) * self._flags.init_learning_rate
                optimizer = fluid.optimizer.AdamOptimizer(
                    learning_rate=learning_rate, beta1=self._flags.adam_beta1,
                    beta2=self._flags.adam_beta2, epsilon=self._flags.opt_epsilon)
                logging.info("use noam_decay learning_rate_scheduler for optimizer.")
                net_output["optimizer"] = optimizer

            net_output["loss"] = loss 
            model_output['fetch_targets'] = [inputs["prefix_letter_id"], pos_score]
        else:
            if self._flags.dump_vec == "query":
                model_output['fetch_targets'] = [prefix_vec]
                pred_input_keys = pred_input_keys[:query_key_num]
            elif self._flags.dump_vec == "poi":
                model_output['fetch_targets'] = [prefix_vec, pos_score, pos_vec]
            else:
                model_output['fetch_targets'] = [inputs["prefix_letter_id"], pos_score, inputs["label"]]

        model_output['feeded_var_names'] = pred_input_keys 
        
        return net_output
 
    def _get_query_vec(self, inputs):
        """
        get query & user vec
        """
        if self._flags.use_personal: 
            #user_tag_id
            #embedding layer
            tag_emb = fluid.layers.embedding(input=inputs['user_tag_id'], is_sparse=True,
                    size=[self._flags.tag_size, self._flags.emb_dim],
                    param_attr=fluid.ParamAttr(name="tagid_embedding", learning_rate=self._flags.emb_lr),
                    padding_idx=0)
            tag_emb = fluid.layers.sequence_pool(tag_emb, pool_type="sum")
            
            user_clk_geoid = fluid.layers.reshape(fluid.layers.cast(inputs['user_clk_geoid'],
                        dtype="float32"), [-1, 40])
            user_resident_geoid = fluid.layers.reshape(fluid.layers.cast(inputs['user_resident_geoid'],
                        dtype="float32"), [-1, 40])
            user_profile = fluid.layers.cast(inputs['user_navi_drive'], dtype="float32")
            user_pool = fluid.layers.concat([tag_emb, user_clk_geoid, user_resident_geoid, user_profile],
                    axis=1)
            #fc layer
            user_vec = fluid.layers.fc(input=user_pool, size=self._flags.emb_dim, act="leaky_relu",
                    param_attr=fluid.ParamAttr(name='user_fc_weight'),
                    bias_attr=fluid.ParamAttr(name='user_fc_bias'))
            #fluid.layers.Print(user_vec) 
 
        loc_vec = fluid.layers.reshape(fluid.layers.cast(x=inputs['prefix_loc_geoid'],
                    dtype="float32"), [-1, 40])
              
        if self._flags.model_type == "bilstm_net":
            network = self.bilstm_net
        elif self._flags.model_type == "bow_net":
            network = self.bow_net
        elif self._flags.model_type == "cnn_net":
            network = self.cnn_net
        elif self._flags.model_type == "lstm_net":
            network = self.lstm_net
        elif self._flags.model_type == "gru_net":
            network = self.gru_net
        else:
            raise ValueError("Unknown network type!")

        prefix_letter_pool = network(inputs["prefix_letter_id"],
                            "wordid_embedding",
                            self._flags.vocab_size,
                            self._flags.emb_dim,
                            hid_dim=self.hid_dim,
                            fc_dim=0,
                            emb_lr=self._flags.emb_lr)
        if self._flags.use_attention:
            #max-pooling
            prefix_letter_pool = fluid.layers.sequence_pool(prefix_letter_pool, pool_type="max")
       
        prefix_vec = prefix_letter_pool
        if self._flags.prefix_word_id:
            prefix_word_pool = network(inputs["prefix_word_id"],
                                "wordid_embedding",
                                self._flags.vocab_size,
                                self._flags.emb_dim,
                                hid_dim=self.hid_dim,
                                fc_dim=0,
                                emb_lr=self._flags.emb_lr)
            if self._flags.use_attention:
               #max-pooling
               prefix_word_pool = fluid.layers.sequence_pool(prefix_word_pool, pool_type="max")
            prefix_pool = fluid.layers.concat([prefix_letter_pool, prefix_word_pool], axis=1)
            prefix_vec = fluid.layers.fc(input=prefix_pool, size=self.hid_dim, act="leaky_relu",
                    param_attr=fluid.ParamAttr(name='prefix_fc_weight'),
                    bias_attr=fluid.ParamAttr(name='prefix_fc_bias'))
        #vector layer
        #fluid.layers.Print(inputs["prefix_letter_id"])
        #fluid.layers.Print(inputs["prefix_word_id"])
        #fluid.layers.Print(prefix_vec)
        
        if self._flags.use_personal:
            #prefix_time_id
            time_emb = fluid.layers.embedding(input=inputs['prefix_time_id'], is_sparse=True,
                    size=[self._flags.time_size, self._flags.emb_dim],
                    param_attr=fluid.ParamAttr(name="timeid_embedding", learning_rate=self._flags.emb_lr))
            time_emb = fluid.layers.sequence_pool(time_emb, pool_type="sum")
            context_pool = fluid.layers.concat([prefix_vec, loc_vec, time_emb, user_vec], axis=1)
        else:
            context_pool = fluid.layers.concat([prefix_vec, loc_vec], axis=1)
        context_vec = fluid.layers.fc(input=context_pool, size=self._flags.fc_dim, act=self._flags.activate,
                param_attr=fluid.ParamAttr(name='context_fc_weight'),
                bias_attr=fluid.ParamAttr(name='context_fc_bias'))
        return context_vec, context_pool

    def _get_poi_vec(self, inputs, tag):
        """
            get poi vec
            context layer: same with query
            feature extract layer: same with query, same kernal params
            vector layer: fc 
        """
        name_letter_pool = self.cnn_net(inputs[tag + "_name_letter_id"],
                            "wordid_embedding",
                            self._flags.vocab_size,
                            self._flags.emb_dim,
                            hid_dim=self.hid_dim,
                            fc_dim=0,
                            emb_lr=self._flags.emb_lr)
        
        name_word_pool = self.cnn_net(inputs[tag + "_name_word_id"],
                            "wordid_embedding",
                            self._flags.vocab_size,
                            self._flags.emb_dim,
                            hid_dim=self.hid_dim,
                            fc_dim=0,
                            emb_lr=self._flags.emb_lr)
        
        addr_letter_pool = self.cnn_net(inputs[tag + "_addr_letter_id"],
                            "wordid_embedding",
                            self._flags.vocab_size,
                            self._flags.emb_dim,
                            hid_dim=self.hid_dim,
                            fc_dim=0,
                            emb_lr=self._flags.emb_lr)

        addr_word_pool = self.cnn_net(inputs[tag + "_addr_word_id"],
                            "wordid_embedding",
                            self._flags.vocab_size,
                            self._flags.emb_dim,
                            hid_dim=self.hid_dim,
                            fc_dim=0,
                            emb_lr=self._flags.emb_lr)
         
        #fc layer
        loc_vec = fluid.layers.reshape(fluid.layers.cast(x=inputs[tag + '_loc_geoid'],
                    dtype="float32"), [-1, 40])
         
        if self._flags.use_attention:
            addr2name_letter_att = dot_product_attention(name_letter_pool, addr_letter_pool,
                    addr_letter_pool, self.hid_dim)
            name2addr_letter_att = dot_product_attention(addr_letter_pool, name_letter_pool,
                    name_letter_pool, self.hid_dim)
            letter_vec = fluid.layers.sequence_concat([addr2name_letter_att, name2addr_letter_att])
            letter_att = ffn(letter_vec, self.hid_dim, self.hid_dim, "inter_ffn")
            #max-pooling
            name_vec = fluid.layers.sequence_pool(letter_att, pool_type="max")

            addr2name_word_att = dot_product_attention(name_word_pool, addr_word_pool,
                    addr_word_pool, self.hid_dim)
            name2addr_word_att = dot_product_attention(addr_word_pool, name_word_pool,
                    name_word_pool, self.hid_dim)
            word_vec = fluid.layers.sequence_concat([addr2name_word_att, name2addr_word_att])
            word_att = ffn(word_vec, self.hid_dim, self.hid_dim, "inter_ffn") 
            #max-pooling
            addr_vec = fluid.layers.sequence_pool(word_att, pool_type="max")
        else:
            name_pool = fluid.layers.concat([name_letter_pool, name_word_pool], axis=1)
            name_vec = fluid.layers.fc(input=name_pool, size=self.hid_dim, act="leaky_relu", 
                    param_attr=fluid.ParamAttr(name='name_fc_weight'),
                    bias_attr=fluid.ParamAttr(name='name_fc_bias'))
            addr_pool = fluid.layers.concat([addr_letter_pool, addr_word_pool], axis=1)
            addr_vec = fluid.layers.fc(input=addr_pool, size=self.hid_dim, act="leaky_relu", 
                    param_attr=fluid.ParamAttr(name='addr_fc_weight'),
                    bias_attr=fluid.ParamAttr(name='addr_fc_bias'))
        
        if self._flags.use_personal: 
            tag_emb = fluid.layers.embedding(input=inputs[tag + '_tag_id'], is_sparse=True,
                    size=[self._flags.tag_size, self._flags.emb_dim],
                    param_attr=fluid.ParamAttr(name="tagid_embedding", learning_rate=self._flags.emb_lr),
                    padding_idx=0)
            tag_emb = fluid.layers.sequence_pool(tag_emb, pool_type="sum")
            poi_pool = fluid.layers.concat([name_vec, addr_vec, loc_vec, tag_emb], axis=1)
        else:
            poi_pool = fluid.layers.concat([name_vec, addr_vec, loc_vec], axis=1)
        #vector layer
        #fluid.layers.Print(inputs[tag + "_name_letter_id"])
        #fluid.layers.Print(inputs[tag + "_name_word_id"])
        #fluid.layers.Print(poi_pool)
        poi_vec = fluid.layers.fc(input=poi_pool, size=self._flags.fc_dim, act=self._flags.activate,
                param_attr=fluid.ParamAttr(name='poi_fc_weight'),
                bias_attr=fluid.ParamAttr(name='poi_fc_bias'))

        return poi_vec, poi_pool

    def train_format(self, result, global_step, epoch_id, batch_id):
        """
            result: one batch train narray
        """ 
        if global_step == 0 or global_step % self._flags.log_every_n_steps != 0:
            return
        
        #result[0] default is loss.
        avg_res = np.mean(np.array(result[0]))
        vec = []
        for i in range(1, len(result)):
            res = np.array(result[i])
            vec.append("%s#%s" % (res.shape, ' '.join(str(j) for j in res.flatten())))
        logging.info("epoch[%s], global_step[%s], batch_id[%s], extra_info: "
                "loss[%s], debug[%s]" % (epoch_id, global_step, batch_id,
                avg_res, ";".join(vec)))

    def init_params(self, place):
        """
            init embed
        """
        def _load_parameter(pretraining_file, vocab_size, word_emb_dim):
            pretrain_word2vec = np.zeros([vocab_size, word_emb_dim], dtype=np.float32)
            for line in open(pretraining_file, 'r'):
                id, _, vec = line.strip('\r\n').split('\t')
                pretrain_word2vec[int(id)] = map(float, vec.split())

                return pretrain_word2vec

        embedding_param = fluid.global_scope().find_var("wordid_embedding").get_tensor()
        pretrain_word2vec = _load_parameter(self._flags.init_train_params,
                self._flags.vocab_size, self._flags.emb_dim)
        embedding_param.set(pretrain_word2vec, place)
        logging.info("init pretrain word2vec:%s" % self._flags.init_train_params)

    def pred_format(self, result, **kwargs):
        """
            format pred output
        """
        if result is None:
            return
    
        if result == '_PRE_':
            if self._flags.dump_vec not in ('query', 'poi'):
                self.idx2word = {} 
                with open(self._flags.qac_dict_path, 'r') as f:
                    for line in f:
                        term, tag, cnt, is_stop, term_id = line.strip('\r\n').split('\t')
                        self.idx2word[int(term_id)] = term
            return

        if result == '_POST_':
            if self._flags.init_pretrain_model is not None:
                path = "%s/infer_model" % (self._flags.export_dir)
                frame_env = kwargs['frame_env']
                fluid.io.save_inference_model(path,
                       frame_env.paddle_env['feeded_var_names'],
                       frame_env.paddle_env['fetch_targets'],
                       frame_env.paddle_env['exe'], frame_env.paddle_env['program'])

            return 

        if self._flags.dump_vec == "query":
            prefix_vec = np.array(result[0])
            for q in prefix_vec:
                print("qid\t%s" % (" ".join(map(str, q))))
        elif self._flags.dump_vec == "poi":
            poi_score = np.array(result[1])
            poi_vec = np.array(result[2])
            for i in range(len(poi_score)):
                print("bid\t%s\t%s" % (poi_score[i][0], " ".join(map(str, poi_vec[i]))))
        else:
            prefix_id = result[0]
            pred_score = np.array(result[1])
            label = np.array(result[2])
            for i in range(len(pred_score)):
                start = prefix_id.lod()[0][i]
                end = prefix_id.lod()[0][i + 1]
                words = []
                for idx in np.array(prefix_id)[start:end]:
                    words.append(self.idx2word.get(idx[0], "UNK"))
                print("qid_%s\t%s\t%s" % ("".join(words), label[i][0], pred_score[i][0]))

    def bow_net(self,
                data,
                layer_name,
                dict_dim,
                emb_dim=128,
                hid_dim=128,
                fc_dim=128, emb_lr=0.1):
        """
        bow net
        """
        # embedding layer
        emb = fluid.layers.embedding(input=data, is_sparse=True, size=[dict_dim, emb_dim],
                param_attr=fluid.ParamAttr(name=layer_name, learning_rate=emb_lr), padding_idx=0)
        
        # bow layer
        bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
        #bow = fluid.layers.tanh(bow)
        #bow = fluid.layers.softsign(bow)
        
        # full connect layer
        if fc_dim > 0:
            bow = fluid.layers.fc(input=bow, size=fc_dim, act=self._flags.activate)
        return bow 
     
    def cnn_net(self,
                data,
                layer_name,
                dict_dim,
                emb_dim=128,
                hid_dim=128,
                fc_dim=96,
                win_size=3, emb_lr=0.1):
        """
        conv net
        """
        # embedding layer
        emb = fluid.layers.embedding(input=data, is_sparse=True, size=[dict_dim, emb_dim],
                param_attr=fluid.ParamAttr(name=layer_name, learning_rate=emb_lr), padding_idx=0)
        
        param_attr = fluid.ParamAttr(
            name="conv_weight",
            initializer=fluid.initializer.TruncatedNormalInitializer(loc=0.0, scale=0.1))
        bias_attr = fluid.ParamAttr(
            name="conv_bias",
            initializer=fluid.initializer.Constant(0.0))
        
        if self._flags.use_attention:
            # convolution layer
            conv = fluid.layers.sequence_conv(
                input=emb,
                num_filters=hid_dim,
                filter_size=win_size,
                param_attr=param_attr,
                bias_attr=bias_attr,
                act="leaky_relu") #tanh
            att = dot_product_attention(conv, conv, conv, hid_dim)
            conv = ffn(att, hid_dim, hid_dim, "intra_ffn")
        else:
            # convolution layer
            conv = fluid.nets.sequence_conv_pool(
                input=emb,
                num_filters=hid_dim,
                filter_size=win_size,
                param_attr=param_attr,
                bias_attr=bias_attr,
                act="leaky_relu", #tanh 
                pool_type="max")
            # full connect layer
            if fc_dim > 0:
                conv = fluid.layers.fc(input=conv, size=fc_dim, act=self._flags.activate)
        return conv
 
    def lstm_net(self, 
                 data,
                 layer_name,
                 dict_dim,
                 emb_dim=128,
                 hid_dim=128,
                 fc_dim=96,
                 emb_lr=0.1):
        """
        lstm net
        """
        # embedding layer
        emb = fluid.layers.embedding(input=data, is_sparse=True, size=[dict_dim, emb_dim],
                param_attr=fluid.ParamAttr(name=layer_name, learning_rate=emb_lr), padding_idx=0)
        
        # Lstm layer
        fc0 = fluid.layers.fc(input=emb, size=hid_dim * 4,
                param_attr=fluid.ParamAttr(name='lstm_fc_weight'),
                bias_attr=fluid.ParamAttr(name='lstm_fc_bias'))
        lstm_h, c = fluid.layers.dynamic_lstm(input=fc0, size=hid_dim * 4, is_reverse=False,
                param_attr=fluid.ParamAttr(name='lstm_weight'),
                bias_attr=fluid.ParamAttr(name='lstm_bias'))
        # max pooling layer
        lstm = fluid.layers.sequence_pool(input=lstm_h, pool_type='max')
        lstm = fluid.layers.tanh(lstm)
    
        # full connect layer
        if fc_dim > 0:
            lstm = fluid.layers.fc(input=lstm, size=fc_dim, act=self._flags.activate)
        return lstm

    def bilstm_net(self,
                   data,
                   layer_name,
                   dict_dim,
                   emb_dim=128,
                   hid_dim=128,
                   fc_dim=96,
                   emb_lr=0.1):
        """
        bi-Lstm net
        """
        # embedding layer
        emb = fluid.layers.embedding(input=data, is_sparse=True, size=[dict_dim, emb_dim],
                param_attr=fluid.ParamAttr(name=layer_name, learning_rate=emb_lr), padding_idx=0)
        
        #LSTM layer
        fc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)
        rfc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)
        lstm_h, c = fluid.layers.dynamic_lstm(input=fc0, size=hid_dim * 4, is_reverse=False)
        rlstm_h, c = fluid.layers.dynamic_lstm(input=rfc0, size=hid_dim * 4, is_reverse=True)
        # extract last layer
        lstm_last = fluid.layers.sequence_last_step(input=lstm_h)
        rlstm_last = fluid.layers.sequence_last_step(input=rlstm_h)
        #lstm_last = fluid.layers.tanh(lstm_last)
        #rlstm_last = fluid.layers.tanh(rlstm_last)
        # concat layer
        bi_lstm = fluid.layers.concat(input=[lstm_last, rlstm_last], axis=1)
        
        # full connect layer
        if fc_dim > 0:
            bi_lstm = fluid.layers.fc(input=bi_lstm, size=fc_dim, act=self._flags.activate)
        return bi_lstm 
      
    def gru_net(self,
                data,
                layer_name,
                dict_dim,
                emb_dim=128,
                hid_dim=128,
                fc_dim=96,
                emb_lr=0.1):
        """
        gru net
        """
        emb = fluid.layers.embedding(input=data, is_sparse=True, size=[dict_dim, emb_dim],
                param_attr=fluid.ParamAttr(name=layer_name, learning_rate=emb_lr), padding_idx=0)
       
        #gru layer
        fc0 = fluid.layers.fc(input=emb, size=hid_dim * 3)
        gru = fluid.layers.dynamic_gru(input=fc0, size=hid_dim, is_reverse=False)
        gru = fluid.layers.sequence_pool(input=gru, pool_type='max')
        #gru = fluid.layers.tanh(gru)
        
        if fc_dim > 0:
            gru = fluid.layers.fc(input=gru, size=fc_dim, act=self._flags.activate)
        return gru

     
