#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""define network paradigm"""

import sys
import re

import paddle
import paddle.fluid as fluid


class Paradigm(object): 
    """
    define network paradigm
    """
    def __init__(self, task_name): 
        """
        init
        """
        self.task_name = task_name

    def create_cls(self, transformer_inst, params): 
        """
        create classify paradigm network
        """
        cls_feats = transformer_inst.get_pooled_output()
        cls_feats = fluid.layers.dropout(
            x=cls_feats,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")
        logits = fluid.layers.fc(
            input=cls_feats,
            size=params['num_labels'],
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.)))

        if params['is_prediction']: 
            probs = fluid.layers.softmax(logits)
            feed_targets_name = [
                params['src_ids'].name, params['pos_ids'].name, 
                params['sent_ids'].name, params['input_mask'].name,
                ]
            results = {"probs": probs, 
                       "feed_targets_name": feed_targets_name}
            return results

        ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
            logits=logits, label=params['labels'], return_softmax=True)
        loss = fluid.layers.reduce_mean(input=ce_loss)
        num_seqs = fluid.layers.create_tensor(dtype='int64')
        accuracy = fluid.layers.accuracy(input=probs, label=params['labels'], total=num_seqs)

        loss.persistable = True
        probs.persistable = True
        accuracy.persistable = True
        num_seqs.persistable = True

        results = {"loss": loss, 
                   "probs": probs, 
                   "accuracy": accuracy, 
                   "num_seqs": num_seqs}
        return results

    def create_multi_cls(self, transformer_inst, params): 
        """
        create multi classify paradigm network
        """
        cls_feats = transformer_inst.get_pooled_output()
        cls_feats = fluid.layers.dropout(
                     x=cls_feats,
                     dropout_prob=0.1,
                     dropout_implementation="upscale_in_train")
        logits = fluid.layers.fc(
            input=cls_feats,
            size=params['num_labels'],
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.)))

        labels_onehot = fluid.layers.cast(params["labels"], dtype='float32')
        ce_loss = fluid.layers.reduce_sum(fluid.layers.sigmoid_cross_entropy_with_logits(x=logits, label=labels_onehot))
        loss = fluid.layers.reduce_mean(input=ce_loss)
        probs = fluid.layers.sigmoid(logits)

        if params['is_prediction']: 
            feed_targets_name = [
                params['src_ids'].name, params['pos_ids'].name,
                params['sent_ids'].name, params['input_mask'].name,
                ]
            results = {"probs": probs, 
                       "feed_targets_name": feed_targets_name}
            return results

        num_seqs = fluid.layers.tensor.fill_constant(shape=[1], dtype='int64', value=1)

        loss.persistable = True
        probs.persistable = True
        num_seqs.persistable = True
        results = {"loss": loss, 
                   "probs": probs, 
                   "num_seqs": num_seqs}
        return results

    def create_sequence_tagging(self, transformer_inst, params): 
        """
        create sequence tagging paradigm
        """
        output_layer = transformer_inst.get_sequence_output()
        hidden_size = output_layer.shape[-1]
        output_layer = fluid.layers.stack(output_layer, axis=1)
        output_layer = fluid.layers.reshape(output_layer, [-1, hidden_size])

        logits = fluid.layers.fc(input=output_layer, size=params['num_labels'])
        probs = fluid.layers.cast(fluid.layers.argmax(logits, axis=1), dtype='int32')

        if params['is_prediction']: 
            feed_targets_name = [
                params['src_ids'].name, params['pos_ids'].name,
                params['sent_ids'].name, params['input_mask'].name,
                ]
            results = {"probs": probs, 
                       "feed_targets_name": feed_targets_name}
            return results
      
        num_seqs = fluid.layers.tensor.fill_constant(shape=[1], dtype='int64', value=1)
        y_label_reshape = fluid.layers.cast(fluid.layers.reshape(params['labels'], [-1]), dtype='int32')
        correct_prediction = fluid.layers.equal(probs, y_label_reshape)
        accuracy = fluid.layers.reduce_mean(fluid.layers.cast(correct_prediction, dtype='float32'))
        ce_loss = fluid.layers.softmax_with_cross_entropy(logits=logits, \
                label=fluid.layers.reshape(params['labels'], [-1, 1]))
        loss = fluid.layers.reduce_mean(input=ce_loss)
  
        loss.persistable = True
        probs.persistable = True
        accuracy.persistable = True
        num_seqs.persistable = True
        results = {"loss": loss, 
                   "probs": probs, 
                   "accuracy": accuracy, 
                   "num_seqs": num_seqs}
        return results

    def paradigm(self, transformer_inst, params): 
        """
        run paradigm
        """
        results = None
        if self.task_name == 'udc': 
            results = self.create_cls(transformer_inst, params)
        elif self.task_name == 'swda':
            results = self.create_cls(transformer_inst, params)
        elif self.task_name == 'mrda': 
            results = self.create_cls(transformer_inst, params)
        elif self.task_name == 'atis_intent': 
            results = self.create_cls(transformer_inst, params)
        elif self.task_name == 'atis_slot': 
            results = self.create_sequence_tagging(transformer_inst, params)
        elif self.task_name == 'dstc2':
            results = self.create_multi_cls(transformer_inst, params)
        return results


