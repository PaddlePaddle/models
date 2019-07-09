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
"""Model for classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import paddle.fluid as fluid

sys.path.append("./BERT")
from model.bert import BertModel


def create_model(args,
                 pyreader_name,
                 bert_config,
                 num_labels,
                 is_prediction=False):
    """
    define fine-tuning model
    """
    if args.binary:
        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                    [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                    [-1, 1], [-1, 1]],
            dtypes=['int64', 'int64', 'int64', 'float32', 'int64', 'int64'],
            lod_levels=[0, 0, 0, 0, 0, 0],
            name=pyreader_name,
            use_double_buffer=True)

    (src_ids, pos_ids, sent_ids, input_mask, seq_len,
     labels) = fluid.layers.read_file(pyreader)

    bert = BertModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        config=bert_config,
        use_fp16=args.use_fp16)

    if args.sub_model_type == 'raw':
        cls_feats = bert.get_pooled_output()

    elif args.sub_model_type == 'cnn':
        bert_seq_out = bert.get_sequence_output()
        bert_seq_out = fluid.layers.sequence_unpad(bert_seq_out, seq_len)
        cnn_hidden_size = 100
        convs = []
        for h in [3, 4, 5]:
            conv_feats = fluid.layers.sequence_conv(
                input=bert_seq_out, num_filters=cnn_hidden_size, filter_size=h)
            conv_feats = fluid.layers.batch_norm(input=conv_feats, act="relu")
            conv_feats = fluid.layers.sequence_pool(
                input=conv_feats, pool_type='max')
            convs.append(conv_feats)

        cls_feats = fluid.layers.concat(input=convs, axis=1)

    elif args.sub_model_type == 'gru':
        bert_seq_out = bert.get_sequence_output()
        bert_seq_out = fluid.layers.sequence_unpad(bert_seq_out, seq_len)
        gru_hidden_size = 1024
        gru_input = fluid.layers.fc(input=bert_seq_out,
                                    size=gru_hidden_size * 3)
        gru_forward = fluid.layers.dynamic_gru(
            input=gru_input, size=gru_hidden_size, is_reverse=False)
        gru_backward = fluid.layers.dynamic_gru(
            input=gru_input, size=gru_hidden_size, is_reverse=True)
        gru_output = fluid.layers.concat([gru_forward, gru_backward], axis=1)
        cls_feats = fluid.layers.sequence_pool(
            input=gru_output, pool_type='max')

    elif args.sub_model_type == 'ffa':
        bert_seq_out = bert.get_sequence_output()
        attn = fluid.layers.fc(input=bert_seq_out,
                               num_flatten_dims=2,
                               size=1,
                               act='tanh')
        attn = fluid.layers.softmax(attn)
        weighted_input = bert_seq_out * attn
        weighted_input = fluid.layers.sequence_unpad(weighted_input, seq_len)
        cls_feats = fluid.layers.sequence_pool(weighted_input, pool_type='sum')

    else:
        raise NotImplementedError("%s is not implemented!" %
                                  args.sub_model_type)

    cls_feats = fluid.layers.dropout(
        x=cls_feats,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")

    logits = fluid.layers.fc(
        input=cls_feats,
        size=num_labels,
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)))
    probs = fluid.layers.softmax(logits)

    if is_prediction:
        feed_targets_name = [
            src_ids.name, pos_ids.name, sent_ids.name, input_mask.name
        ]
        return pyreader, probs, feed_targets_name

    ce_loss = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=labels)
    loss = fluid.layers.mean(x=ce_loss)

    if args.use_fp16 and args.loss_scaling > 1.0:
        loss *= args.loss_scaling

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)

    return (pyreader, loss, probs, accuracy, labels, num_seqs)
