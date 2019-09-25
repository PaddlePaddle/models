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


def compute_loss(output_tensors, args=None):
    """Compute loss for mrc model"""
    def _compute_single_loss(logits, positions):
        """Compute start/end loss for mrc model"""
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=logits, label=positions)
        loss = fluid.layers.mean(x=loss)
        return loss

    start_logits = output_tensors['start_logits']
    end_logits = output_tensors['end_logits']
    start_positions = output_tensors['start_positions']
    end_positions = output_tensors['end_positions']
    start_loss = _compute_single_loss(start_logits, start_positions)
    end_loss = _compute_single_loss(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2.0
    if args.use_fp16 and args.loss_scaling > 1.0:
        total_loss = total_loss * args.loss_scaling

    return total_loss


def compute_distill_loss(output_tensors, args=None): 
    """Compute loss for mrc model"""
    start_logits = output_tensors['start_logits']
    end_logits = output_tensors['end_logits']
    start_logits_truth = output_tensors['start_logits_truth']
    end_logits_truth = output_tensors['end_logits_truth']
    input_mask = output_tensors['input_mask']
    def _mask(logits, input_mask, nan=1e5):
        input_mask = fluid.layers.reshape(input_mask, [-1, 512])
        logits = logits - (1.0 - input_mask) * nan
        return logits
    start_logits = _mask(start_logits, input_mask)
    end_logits = _mask(end_logits, input_mask)
    start_logits_truth = _mask(start_logits_truth, input_mask)
    end_logits_truth = _mask(end_logits_truth, input_mask)
    start_logits_truth = fluid.layers.reshape(start_logits_truth, [-1, 512])
    end_logits_truth = fluid.layers.reshape(end_logits_truth, [-1, 512])
    T = 1.0
    start_logits_softmax = fluid.layers.softmax(input=start_logits/T)
    end_logits_softmax = fluid.layers.softmax(input=end_logits/T)
    start_logits_truth_softmax = fluid.layers.softmax(input=start_logits_truth/T)
    end_logits_truth_softmax = fluid.layers.softmax(input=end_logits_truth/T)
    start_logits_truth_softmax.stop_gradient = True
    end_logits_truth_softmax.stop_gradient = True
    start_loss = fluid.layers.cross_entropy(start_logits_softmax, start_logits_truth_softmax, soft_label=True)
    end_loss = fluid.layers.cross_entropy(end_logits_softmax, end_logits_truth_softmax, soft_label=True)
    start_loss = fluid.layers.mean(x=start_loss)
    end_loss = fluid.layers.mean(x=end_loss)
    total_loss = (start_loss + end_loss) / 2.0
    return total_loss


def create_model(reader_input, base_model=None, is_training=True, args=None):
    """
        given the base model, reader_input
        return the output tensors
    """

    if is_training: 
        if args.do_distill: 
            src_ids, pos_ids, sent_ids, input_mask, \
                start_logits_truth, end_logits_truth, start_positions, end_positions = reader_input
        else: 
            src_ids, pos_ids, sent_ids, input_mask, \
                start_positions, end_positions = reader_input
    else:
        src_ids, pos_ids, sent_ids, input_mask, unique_id = reader_input
    enc_out = base_model.get_output("sequence_output")
    logits = fluid.layers.fc(
        input=enc_out,
        size=2,
        num_flatten_dims=2,
        param_attr=fluid.ParamAttr(
            name="cls_squad_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_squad_out_b", initializer=fluid.initializer.Constant(0.)))

    logits = fluid.layers.transpose(x=logits, perm=[2, 0, 1])
    start_logits, end_logits = fluid.layers.unstack(x=logits, axis=0)

    batch_ones = fluid.layers.fill_constant_batch_size_like(
        input=start_logits, dtype='int64', shape=[1], value=1)
    num_seqs = fluid.layers.reduce_sum(input=batch_ones)

    output_tensors = {}
    output_tensors['start_logits'] = start_logits
    output_tensors['end_logits'] = end_logits
    output_tensors['num_seqs'] = num_seqs
    output_tensors['input_mask'] = input_mask
    if is_training:
        output_tensors['start_positions'] = start_positions
        output_tensors['end_positions'] = end_positions
        if args.do_distill: 
            output_tensors['start_logits_truth'] = start_logits_truth
            output_tensors['end_logits_truth'] = end_logits_truth

    else:
        output_tensors['unique_id'] = unique_id
        output_tensors['start_logits'] = start_logits
        output_tensors['end_logits'] = end_logits

    return output_tensors
