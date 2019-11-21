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
"""
Generator class.
"""

import bisect
import math
import sys

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.framework import Variable

from plato.args import str2bool
import plato.modules.functions as F


def repeat(var, times):
    if isinstance(var, list):
        return [repeat(x, times) for x in var]
    elif isinstance(var, dict):
        return {k: repeat(v, times) for k, v in var.items()}
    elif isinstance(var, Variable):
        var = F.unsqueeze(var, [1])
        expand_times = [1] * len(var.shape)
        expand_times[1] = times
        dtype = var.dtype
        var = layers.cast(var, "float32")
        var = layers.expand(var, expand_times)
        shape = [var.shape[0] * var.shape[1]] + var.shape[2:]
        var = layers.reshape(var, shape)
        var = layers.cast(var, dtype)
        return var
    else:
        return var


def gather(var, idx):
    if isinstance(var, list):
        return [gather(x, idx) for x in var]
    elif isinstance(var, dict):
        return {k: gather(v, idx) for k, v in var.items()}
    elif isinstance(var, Variable):
        out = layers.gather(var, idx)
        return out
    else:
        return var


class Generator(object):
    """ Genrator class. """

    _registry = dict()

    @classmethod
    def register(cls, name):
        Generator._registry[name] = cls
        return

    @staticmethod
    def by_name(name):
        return Generator._registry[name]

    @staticmethod
    def create(hparams, *args, **kwargs):
        """ Create generator. """
        generator_cls = Generator.by_name(hparams.generator)
        return generator_cls(hparams, *args, **kwargs)

    @classmethod
    def add_cmdline_argument(cls, parser):
        group = parser.add_argument_group("Generator")
        group.add_argument("--generator", type=str, default="BeamSearch",
                           choices=["TopKSampling", "TopPSampling", "GreedySampling",
                                    "BeamSearch"])
        group.add_argument("--min_gen_len", type=int, default=1,
                           help="The minimum length of generated response.")
        group.add_argument("--max_gen_len", type=int, default=30,
                           help="The maximum length of generated response.")
        args, _ = parser.parse_known_args()
        generator_cls = cls.by_name(args.generator)
        generator_cls.add_cmdline_argument(group)
        return group

    def __init__(self, hparams, bpe):
        self.vocab_size = bpe.vocab_size
        self.bos_id = bpe.bos_id
        self.eos_id = bpe.eos_id
        self.unk_id = bpe.unk_id
        self.pad_id = bpe.pad_id
        self.min_gen_len = hparams.min_gen_len
        self.max_gen_len = hparams.max_gen_len
        assert 1 <= self.min_gen_len <= self.max_gen_len
        return

    def __call__(self, step_fn, state):
        """
        Running generation.

        @param : step_fn : decoding one step
        @type : function

        @param : state : initial state
        @type : dict
        """
        raise NotImplementedError


class Sampling(Generator):
    """ Sampling Generator. """

    @classmethod
    def add_cmdline_argument(cls, group):
        group.add_argument("--ignore_unk", type=str2bool, default=True,
                           help="Whether to ignore unkown token in generation.")
        group.add_argument("--sampling_temperature", type=float, default=1.0)
        return group

    def __init__(self, hparams, bpe):
        super().__init__(hparams, bpe)
        self.ignore_unk = hparams.ignore_unk
        self.temperature = hparams.sampling_temperature
        return

    def _sampling(self, scores):
        """ Sampling function. """
        raise NotImplementedError

    def __call__(self, step_fn, state):
        """
        Running generation.

        @param : step_fn : decoding one step
        @type : function

        @param : state : initial state
        @type : dict
        """
        batch_size = state["batch_size"]
        vocab_size = self.vocab_size

        pos_index = layers.range(0, batch_size, 1, dtype="int64")
        pos_index = layers.scale(pos_index, vocab_size)

        # shape: [batch_size, beam_size, 1]
        predictions = layers.fill_constant(shape=[batch_size, 1],
                                           dtype="int64",
                                           value=self.bos_id)
        sequence_scores = layers.fill_constant(shape=[batch_size],
                                               dtype="float32",
                                               value=0.0)

        unk_penalty = np.zeros(vocab_size, dtype="float32")
        unk_penalty[self.unk_id] = -1e10
        unk_penalty = layers.assign(unk_penalty)

        eos_penalty = np.zeros(vocab_size, dtype="float32")
        eos_penalty[self.eos_id] = -1e10
        eos_penalty = layers.assign(eos_penalty)

        scores_after_end = np.full(vocab_size, -1e10, dtype="float32")
        scores_after_end[self.pad_id] = 0
        scores_after_end = layers.assign(scores_after_end)

        # initial input
        for step in range(1, self.max_gen_len + 1):
            pre_ids = predictions[:, -1:]
            state["pred_token"] = F.unsqueeze(pre_ids, [2])
            if step > 1:
                state["pred_mask"] = 1 - F.equal(state["pred_token"], self.pad_id)
                state["pred_pos"] = state["pred_pos"] + 1
            scores, state = step_fn(state)

            # Generate next
            # scores shape: [batch_size, vocab_size]
            if self.ignore_unk:
                scores = scores + unk_penalty

            if step <= self.min_gen_len:
                scores = scores + eos_penalty

            # previous token is [PAD] or [EOS]
            # shape: [batch_size, 1]
            pre_eos_mask = F.equal(pre_ids, self.eos_id) + F.equal(pre_ids, self.pad_id)
            scores = scores * (1 - pre_eos_mask) + \
                layers.expand(pre_eos_mask, [1, vocab_size]) * scores_after_end

            scores = scores / self.temperature
            preds = self._sampling(scores)

            predictions = layers.concat([predictions, F.unsqueeze(preds, [1])], axis=1)

            scores = layers.reshape(scores, [batch_size * vocab_size])
            preds = preds + pos_index
            scores = gather(scores, preds)
            sequence_scores = sequence_scores + scores

        results = {
            "preds": predictions,
            "scores": sequence_scores
        }
        return results


class GreedySampling(Sampling):
    """ Greedy sampling. """

    @classmethod
    def add_cmdline_argument(cls, group):
        return Sampling.add_cmdline_argument(group)

    def _sampling(self, logits):
        """ Implement greedy sampling. """
        preds = layers.argmax(logits, axis=1)
        return preds


class TopKSampling(Sampling):
    """ Top-k sampling. """

    @classmethod
    def add_cmdline_argument(cls, group):
        Sampling.add_cmdline_argument(group)
        group.add_argument("--top_k_ratio", type=float, default=None)
        group.add_argument("--top_k_num", type=int, default=None)
        return group

    def __init__(self, hparams, bpe):
        super().__init__(hparams, bpe)
        assert hparams.top_k_ratio is not None or hparams.top_k_num is not None
        if hparams.top_k_num is not None:
            self.top_k_num = hparams.top_k_num
        else:
            self.top_k_num = math.floor(hparams.top_k_ratio * self.vocab_size)
        assert self.top_k_num >= 1
        return

    def _sampling(self, logits):
        """ Implement top-k sampling. """
        probs = layers.softmax(logits, axis=1)
        probs, indices = layers.topk(probs, self.top_k_num)
        probs = probs / layers.reduce_sum(probs, dim=1, keep_dim=True)
        preds = []
        for p, ids in zip(probs.numpy(), indices.numpy()):
            o = np.random.choice(ids, p=p)
            preds.append(o)
        preds = np.array(preds, dtype="int64")
        return fluid.dygraph.to_variable(preds)


class TopPSampling(Sampling):
    """ Top-p sampling. """

    @classmethod
    def add_cmdline_argument(cls, group):
        Sampling.add_cmdline_argument(group)
        group.add_argument("--top_p_ratio", type=float, default=1.0)
        return group

    def __init__(self, hparams, bpe):
        super().__init__(hparams, bpe)
        self.top_p_ratio = hparams.top_p_ratio
        return

    def _sampling(self, logits):
        """ Implement top-k sampling. """
        probs = layers.softmax(logits, axis=1)
        preds = []
        for p in probs.numpy():
            ids = np.argsort(-p)
            p = p[ids]
            c_p = np.cumsum(p)
            i = bisect.bisect_right(c_p, self.top_p_ratio) + 1
            o = np.random.choice(ids[:i], p=p[:i]/np.sum(p[:i]))
            preds.append(o)
        preds = np.array(preds, dtype="int64")
        return fluid.dygraph.to_variable(preds)


class BeamSearch(Generator):
    """ BeamSearch generator. """

    @classmethod
    def add_cmdline_argument(cls, group):
        group.add_argument("--beam_size", type=int, default=5,
                           help="The beam size in beam search.")
        group.add_argument("--length_average", type=str2bool, default=False,
                           help="Whether to use length average.")
        group.add_argument("--length_penalty", type=float, default=-1.0,
                           help="The parameter(alpha) of length penalty.")
        group.add_argument("--ignore_unk", type=str2bool, default=True,
                           help="Whether to ignore unkown token in generation.")
        return group

    def __init__(self, hparams, bpe):
        super().__init__(hparams, bpe)
        self.beam_size = hparams.beam_size
        self.length_average = hparams.length_average
        self.length_penalty = hparams.length_penalty
        self.ignore_unk = hparams.ignore_unk
        return

    def __call__(self, step_fn, state):
        """
        Running beam search.

        @param : step_fn : decoding one step
        @type : function

        @param : state : initial state
        @type : dict
        """
        batch_size = state["batch_size"]
        beam_size = self.beam_size

        # shape: [batch_size, 1]
        pos_index = layers.range(0, batch_size, 1, dtype="int64")
        pos_index = layers.scale(pos_index, beam_size)
        pos_index = F.unsqueeze(pos_index, [1])

        # shape: [batch_size, beam_size, 1]
        predictions = layers.fill_constant(shape=[batch_size, beam_size, 1],
                                           dtype="int64",
                                           value=self.bos_id)

        # initial input
        state["pred_token"] = predictions[:, :1]
        # shape: [batch_size, vocab_size]
        scores, state = step_fn(state)

        unk_penalty = np.zeros(self.vocab_size, dtype="float32")
        unk_penalty[self.unk_id] = -1e10
        unk_penalty = layers.assign(unk_penalty)

        eos_penalty = np.zeros(self.vocab_size, dtype="float32")
        eos_penalty[self.eos_id] = -1e10
        eos_penalty = layers.assign(eos_penalty)

        scores_after_end = np.full(self.vocab_size, -1e10, dtype="float32")
        scores_after_end[self.pad_id] = 0
        scores_after_end = layers.assign(scores_after_end)

        if self.ignore_unk:
            scores = scores + unk_penalty
        scores = scores + eos_penalty

        # shape: [batch_size, beam_size]
        sequence_scores, preds = layers.topk(scores, self.beam_size)

        predictions = layers.concat([predictions, F.unsqueeze(preds, [2])], axis=2)
        state = repeat(state, beam_size)

        parent_idx_list = []
        pred_list = []

        for step in range(2, self.max_gen_len + 1):
            pre_ids = predictions[:, :, -1:]
            state["pred_token"] = layers.reshape(pre_ids, shape=[batch_size * beam_size, 1, 1])
            state["pred_mask"] = 1 - F.equal(state["pred_token"], self.pad_id)
            state["pred_pos"] = state["pred_pos"] + 1
            scores, state = step_fn(state)

            # Generate next
            # scores shape: [batch_size, beam_size, vocab_size]
            if self.ignore_unk:
                scores = scores + unk_penalty

            if step <= self.min_gen_len:
                scores = scores + eos_penalty

            scores = layers.reshape(scores, shape=[batch_size, beam_size, self.vocab_size])

            # previous token is [PAD] or [EOS]
            pre_eos_mask = F.equal(pre_ids, self.eos_id) + F.equal(pre_ids, self.pad_id)

            scores = scores * (1 - pre_eos_mask) + \
                layers.expand(pre_eos_mask, [1, 1, self.vocab_size]) * scores_after_end
            if self.length_average:
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * (1 - 1 / step)
                sequence_scores = F.unsqueeze(sequence_scores, [2]) * scaled_value
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * (1 / step)
                scores = scores * scaled_value
            elif self.length_penalty >= 0.0:
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * \
                    (math.pow((4 + step) / (5 + step), self.length_penalty))
                sequence_scores = layers.elementwise_mul(scaled_value, sequence_scores, axis=0)
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * \
                    (math.pow(1 / (5 + step), self.length_penalty))
                scores = scores * scaled_value
            scores = layers.elementwise_add(scores, sequence_scores, axis=0)
            scores = layers.reshape(scores, shape=[batch_size, beam_size * self.vocab_size])

            topk_scores, topk_indices = layers.topk(scores, beam_size)
            vocab_size = layers.fill_constant(shape=[1], dtype="int64", value=self.vocab_size)
            parent_idx = layers.elementwise_floordiv(topk_indices, vocab_size)
            preds = layers.elementwise_mod(topk_indices, vocab_size)

            # Gather state / sequence_scores
            parent_idx = layers.elementwise_add(parent_idx, pos_index, axis=0)
            parent_idx = layers.reshape(parent_idx, [batch_size * beam_size])
            state = gather(state, parent_idx)
            sequence_scores = topk_scores

            predictions = layers.reshape(predictions, shape=[batch_size * beam_size, step])
            predictions = gather(predictions, parent_idx)
            predictions = layers.reshape(predictions, shape=[batch_size, beam_size, step])
            predictions = layers.concat([predictions, F.unsqueeze(preds, [2])], axis=2)

        pre_ids = predictions[:, :, -1]
        pre_eos_mask = F.equal(pre_ids, self.eos_id) + F.equal(pre_ids, self.pad_id)
        sequence_scores = sequence_scores * pre_eos_mask + layers.scale(1 - pre_eos_mask, -1e10)

        _, indices = layers.argsort(sequence_scores, axis=1)
        indices = indices + pos_index
        indices = layers.reshape(indices, [-1])
        sequence_scores = layers.reshape(sequence_scores, [batch_size * beam_size])
        predictions = layers.reshape(predictions, [batch_size * beam_size, -1])
        sequence_scores = gather(sequence_scores, indices)
        predictions = layers.gather(predictions, indices)
        sequence_scores = layers.reshape(sequence_scores, [batch_size, beam_size])
        predictions = layers.reshape(predictions, [batch_size, beam_size, -1])

        results = {
            "preds": predictions[:, -1],
            "scores": sequence_scores[:, -1]
        }
        return results

BeamSearch.register("BeamSearch")
GreedySampling.register("GreedySampling")
TopKSampling.register("TopKSampling")
TopPSampling.register("TopPSampling")
