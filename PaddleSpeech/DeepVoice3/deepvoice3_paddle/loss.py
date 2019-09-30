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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numba import jit

from paddle import fluid
import paddle.fluid.dygraph as dg


def masked_mean(inputs, mask):
    """
    Args:
        inputs (Variable): Shape(B, C, 1, T), the input, where B means
            batch size, C means channels of input, T means timesteps of
            the input.
        mask (Variable): Shape(B, T), a mask. 
    Returns:
        loss (Variable): Shape(1, ), masked mean.
    """
    channels = inputs.shape[1]
    reshaped_mask = fluid.layers.reshape(
        mask, shape=[mask.shape[0], 1, 1, mask.shape[-1]])
    expanded_mask = fluid.layers.expand(
        reshaped_mask, expand_times=[1, channels, 1, 1])
    expanded_mask.stop_gradient = True

    valid_cnt = fluid.layers.reduce_sum(expanded_mask)
    valid_cnt.stop_gradient = True

    masked_inputs = inputs * expanded_mask
    loss = fluid.layers.reduce_sum(masked_inputs) / valid_cnt
    return loss


@jit(nopython=True)
def guided_attention(N, max_N, T, max_T, g):
    W = np.zeros((max_N, max_T), dtype=np.float32)
    for n in range(N):
        for t in range(T):
            W[n, t] = 1 - np.exp(-(n / N - t / T)**2 / (2 * g * g))
    return W


def guided_attentions(input_lengths, target_lengths, max_target_len, g=0.2):
    B = len(input_lengths)
    max_input_len = input_lengths.max()
    W = np.zeros((B, max_target_len, max_input_len), dtype=np.float32)
    for b in range(B):
        W[b] = guided_attention(input_lengths[b], max_input_len,
                                target_lengths[b], max_target_len, g).T
    return W


class TTSLoss(object):
    def __init__(self,
                 masked_weight=0.0,
                 priority_weight=0.0,
                 binary_divergence_weight=0.0,
                 guided_attention_sigma=0.2):
        self.masked_weight = masked_weight
        self.priority_weight = priority_weight
        self.binary_divergence_weight = binary_divergence_weight
        self.guided_attention_sigma = guided_attention_sigma

    def l1_loss(self, prediction, target, mask, priority_bin=None):
        abs_diff = fluid.layers.abs(prediction - target)

        # basic mask-weighted l1 loss
        w = self.masked_weight
        if w > 0 and mask is not None:
            base_l1_loss = w * masked_mean(abs_diff, mask) + (
                1 - w) * fluid.layers.reduce_mean(abs_diff)
        else:
            base_l1_loss = fluid.layers.reduce_mean(abs_diff)

        if self.priority_weight > 0 and priority_bin is not None:
            # mask-weighted priority channels' l1-loss
            priority_abs_diff = fluid.layers.slice(
                abs_diff, axes=[1], starts=[0], ends=[priority_bin])
            if w > 0 and mask is not None:
                priority_loss = w * masked_mean(priority_abs_diff, mask) + (
                    1 - w) * fluid.layers.reduce_mean(priority_abs_diff)
            else:
                priority_loss = fluid.layers.reduce_mean(priority_abs_diff)

            # priority weighted sum
            p = self.priority_weight
            loss = p * priority_loss + (1 - p) * base_l1_loss
        else:
            loss = base_l1_loss
        return loss

    def binary_divergence(self, prediction, target, mask):
        flattened_prediction = fluid.layers.reshape(prediction, [-1, 1])
        flattened_target = fluid.layers.reshape(target, [-1, 1])
        flattened_loss = fluid.layers.log_loss(
            flattened_prediction, flattened_target, epsilon=1e-8)
        bin_div = fluid.layers.reshape(flattened_loss, prediction.shape)

        w = self.masked_weight
        if w > 0 and mask is not None:
            loss = w * masked_mean(bin_div, mask) + (
                1 - w) * fluid.layers.reduce_mean(bin_div)
        else:
            loss = fluid.layers.reduce_mean(bin_div)
        return loss

    @staticmethod
    def done_loss(done_hat, done):
        flat_done_hat = fluid.layers.reshape(done_hat, [-1, 1])
        flat_done = fluid.layers.reshape(done, [-1, 1])
        loss = fluid.layers.log_loss(flat_done_hat, flat_done, epsilon=1e-8)
        loss = fluid.layers.reduce_mean(loss)
        return loss

    def attention_loss(self, predicted_attention, input_lengths,
                       target_lengths):
        """
        Given valid encoder_lengths and decoder_lengths, compute a diagonal 
        guide, and compute loss from the predicted attention and the guide.
        
        Args:
            predicted_attention (Variable): Shape(*, B, T_dec, T_enc), the 
                alignment tensor, where B means batch size, T_dec means number
                of time steps of the decoder, T_enc means the number of time
                steps of the encoder, * means other possible dimensions.
            input_lengths (numpy.ndarray): Shape(B,), dtype:int64, valid lengths
                (time steps) of encoder outputs.
            target_lengths (numpy.ndarray): Shape(batch_size,), dtype:int64, 
                valid lengths (time steps) of decoder outputs.
        
        Returns:
            loss (Variable): Shape(1, ) attention loss.
        """
        n_attention, batch_size, max_target_len, max_input_len = (
            predicted_attention.shape)
        soft_mask = guided_attentions(input_lengths, target_lengths,
                                      max_target_len,
                                      self.guided_attention_sigma)
        soft_mask_ = dg.to_variable(soft_mask)
        loss = fluid.layers.reduce_mean(predicted_attention * soft_mask_)
        return loss
