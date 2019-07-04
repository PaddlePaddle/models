#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import absolute_import

import math
import numpy as np
import paddle.fluid as fluid
import mmpms.layers as layers

from mmpms.models.model_base import Model
from mmpms.modules.embedder import Embedder
from mmpms.modules.encoder import GRUEncoder
from mmpms.modules.decoder import GRUDecoder
from mmpms.utils.misc import sequence_but, sequence_last


class MMPMS(Model):
    def __init__(self, vocab, generator, hparams, optim_hparams, use_gpu=False):
        self.vocab = vocab
        self.generator = generator

        self.vocab_size = self.vocab.size()
        self.embed_dim = hparams.embed_dim
        self.hidden_dim = hparams.hidden_dim
        self.num_mappings = hparams.num_mappings
        self.tau = hparams.tau
        self.num_layers = hparams.num_layers
        self.bidirectional = hparams.bidirectional
        self.attn_mode = hparams.attn_mode
        self.use_pretrained_embedding = hparams.use_pretrained_embedding
        self.embed_init_scale = hparams.embed_init_scale
        self.dropout = hparams.dropout

        self.grad_clip = optim_hparams.grad_clip or 0

        # Embedding
        self.embedder = Embedder(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim,
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                -self.embed_init_scale, self.embed_init_scale)),
            name="embedder")

        # Encoding
        self.post_encoder = GRUEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            name="post_encoder")

        self.response_encoder = GRUEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            name="response_encoder")

        # Multi-Mapping
        self.mappings = layers.LayerList([
            layers.FC(size=self.hidden_dim, name="map_{}".format(i))
            for i in range(self.num_mappings)
        ])

        # Decoding
        self.decoder = GRUDecoder(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            attn_mode=self.attn_mode,
            dropout=self.dropout,
            name="decoder")

        # Predictor
        bound = math.sqrt(1 / self.hidden_dim)
        if self.attn_mode == "none":
            self.predictor = layers.Sequential(
                layers.Dropout(dropout_prob=self.dropout),
                layers.FC(
                    size=self.vocab_size,
                    act="softmax",
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Uniform(-bound, bound)),
                    name="predictor"))
        else:
            self.predictor = layers.Sequential(
                layers.Dropout(dropout_prob=self.dropout),
                layers.FC(size=self.hidden_dim, name="project"),
                layers.FC(
                    size=self.vocab_size,
                    act="softmax",
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Uniform(-bound, bound)),
                    bias_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Uniform(-bound, bound)),
                    name="predictor"), )

        # Optimizer
        Optimizer = getattr(fluid.optimizer, optim_hparams.optimizer)
        self.optimizer = Optimizer(learning_rate=optim_hparams.lr)

        super(MMPMS, self).__init__(use_gpu=use_gpu)

        # Embedding Initialization
        if self.use_pretrained_embedding:
            self.embedder.from_pretrained(self.vocab.embeddings, self.place,
                                          self.embed_init_scale)

    def gumbel_softmax(self, logits, tau, eps=1e-10):
        u = layers.uniform_random_batch_size_like(
            logits, shape=[-1, self.num_mappings], min=0.0, max=1.0)
        u.stop_gradient = True
        gumbel = 0.0 - layers.log(eps - layers.log(u + eps))
        y = logits + gumbel
        return layers.softmax(y / tau)

    def encode(self, post_inputs, response_inputs, is_training=False):
        outputs = {}
        post_enc_inputs = self.embedder(post_inputs)
        post_outputs, post_hidden = self.post_encoder(post_enc_inputs)
        post_hidden = post_hidden[-1]

        # shape: (batch_size, num_mappings, hidden_dim)
        candidate_hiddens = layers.stack(
            [mapping(post_hidden) for mapping in self.mappings], axis=1)

        response_enc_inputs = self.embedder(response_inputs)
        _, response_hidden = self.response_encoder(response_enc_inputs)
        response_hidden = response_hidden[-1]

        # For simplicity, use the target responses in the same batch as negative examples
        neg_response_hidden = layers.reverse(response_hidden, axis=0)

        pos_logits = layers.reduce_sum(
            post_hidden * response_hidden, dim=1, keep_dim=True)
        neg_logits = layers.reduce_sum(
            post_hidden * neg_response_hidden, dim=1, keep_dim=True)
        outputs.update({"pos_logits": pos_logits, "neg_logits": neg_logits})

        # shape: (batch_size, num_mappings)
        similarity = layers.squeeze(
            layers.matmul(
                candidate_hiddens, layers.unsqueeze(
                    response_hidden, axes=[2])),
            axes=[2])
        post_probs = layers.softmax(similarity)
        outputs.update({"post_probs": post_probs})

        if is_training:
            z = self.gumbel_softmax(
                layers.log(post_probs + 1e-10), tau=self.tau)
        else:
            indices = layers.argmax(post_probs, axis=1)
            z = layers.one_hot(
                layers.reshape(
                    indices, shape=[-1, 1]), self.num_mappings)

        # shape: (batch_size, hidden_size)
        dec_hidden = layers.squeeze(
            layers.matmul(
                layers.unsqueeze(
                    z, axes=[1]), candidate_hiddens),
            axes=[1])

        state = {}
        state["hidden"] = [dec_hidden] * self.num_layers
        if self.attn_mode != "none":
            state["memory"] = post_outputs
        return outputs, state

    def enumerate_encode(self, inputs, post_expand_lod):
        post_enc_inputs = self.embedder(inputs)
        post_outputs, post_hidden = self.post_encoder(post_enc_inputs)
        post_hidden = post_hidden[-1]

        # shape: (batch_size*num_mappings, hidden_dim)
        dec_hidden = layers.stack(
            [mapping(post_hidden) for mapping in self.mappings], axis=1)
        dec_hidden = layers.reshape(dec_hidden, shape=[-1, self.hidden_dim])

        post_outputs = layers.expand(
            post_outputs, expand_times=[1, self.num_mappings])
        post_outputs = layers.sequence_reshape(
            post_outputs, new_dim=self.hidden_dim)
        post_outputs = layers.lod_reset(post_outputs, y=post_expand_lod)

        state = {}
        state["hidden"] = [dec_hidden] * self.num_layers
        if self.attn_mode != "none":
            state["memory"] = post_outputs
        return state

    def decode(self, inputs, state, is_infer=True):
        dec_inputs = self.embedder(inputs)
        if is_infer:
            dec_outputs, new_state = self.decoder.step(dec_inputs, state=state)
        else:
            dec_outputs = self.decoder(dec_inputs, state=state)
        probs = self.predictor(dec_outputs)
        if is_infer:
            return probs, new_state
        else:
            return probs

    def collect_metrics(self, outputs, label):
        metrics = {}
        loss = 0

        # Seq2Seq NLL Loss
        probs = outputs["probs"]
        nll = layers.cross_entropy(input=probs, label=label)
        ppl = layers.mean(
            layers.exp(layers.sequence_pool(
                nll, pool_type="average")),
            name="ppl")
        nll = layers.mean(
            layers.sequence_pool(
                nll, pool_type="sum"), name="nll")
        metrics.update({"nll": nll, "ppl": ppl})
        loss += nll

        # Matching Loss
        pos_logits = outputs["pos_logits"]
        pos_label = layers.fill_constant_batch_size_like(
            pos_logits, shape=[-1, 1], dtype="float32", value=1)
        pos_label.stop_gradient = True
        neg_logits = outputs["neg_logits"]
        neg_label = layers.fill_constant_batch_size_like(
            neg_logits, shape=[-1, 1], dtype="float32", value=0)
        neg_label.stop_gradient = True

        pos_loss = layers.sigmoid_cross_entropy_with_logits(pos_logits,
                                                            pos_label)
        neg_loss = layers.sigmoid_cross_entropy_with_logits(neg_logits,
                                                            neg_label)
        match = layers.mean(pos_loss + neg_loss)
        pos_acc = layers.mean(
            layers.cast(
                layers.less_than(neg_label, pos_logits), dtype="float32"))
        neg_acc = layers.mean(
            layers.cast(
                layers.less_than(neg_logits, neg_label), dtype="float32"))
        acc = (pos_acc + neg_acc) / 2.0
        metrics.update({"match": match, "match_acc": acc})
        loss += match

        metrics["loss"] = loss
        return metrics

    def build_program(self):
        self.startup_program = fluid.Program()
        self.train_program = fluid.Program()
        with fluid.program_guard(self.train_program, self.startup_program):
            # Input
            post = layers.data(
                name="post", shape=[1], lod_level=1, dtype="int64")
            response = layers.data(
                name="response", shape=[1], lod_level=1, dtype="int64")
            label = layers.data(
                name="label", shape=[1], lod_level=1, dtype="int64")
            pos_response = layers.data(
                name="pos_response", shape=[1], lod_level=1, dtype="int64")

            self.eval_program = self.train_program.clone(for_test=True)

            # Encode
            outputs, state = self.encode(
                post_inputs=post,
                response_inputs=pos_response,
                is_training=True)

            # Decode
            probs = self.decode(response, state, is_infer=False)
            outputs.update({"probs": probs})

            # Metrics
            metrics = self.collect_metrics(outputs, label)

            loss = metrics["loss"]
            if self.grad_clip > 0:
                fluid.clip.set_gradient_clip(
                    clip=fluid.clip.GradientClipByGlobalNorm(
                        clip_norm=self.grad_clip),
                    program=self.train_program)
            self.optimizer.minimize(loss)
            self.train_fetch_dict = metrics

        with fluid.program_guard(self.eval_program, self.startup_program):
            # Encode
            outputs, state = self.encode(
                post_inputs=post,
                response_inputs=pos_response,
                is_training=False)

            # Decode
            probs = self.decode(response, state, is_infer=False)
            outputs.update({"probs": probs})

            # Metrics
            metrics = self.collect_metrics(outputs, label)
            self.eval_fetch_dict = metrics
        self.eval_program = self.eval_program.clone(for_test=True)

        self.infer_program = fluid.Program()
        with fluid.program_guard(self.infer_program, self.startup_program):
            # Input
            post = layers.data(
                name="post", shape=[1], lod_level=1, dtype="int64")
            response = layers.data(
                name="response", shape=[1], lod_level=1, dtype="int64")
            init_ids = layers.data(
                name="init_ids", shape=[1], lod_level=2, dtype="int64")
            post_expand_lod = layers.data(
                name="post_expand_lod", shape=[1, -1], dtype="int32")

            # Encode
            state = self.enumerate_encode(post, post_expand_lod)

            # Infer
            prediction_ids, prediction_scores = self.generator(self.decode,
                                                               state, init_ids)
        self.infer_program = self.infer_program.clone(for_test=True)
        self.infer_fetch_dict = {
            "preds": prediction_ids,
            "post": post,
            "response": response
        }

    def train(self, inputs, train_state=None):
        return self.execute(
            program=self.train_program,
            feed=self.set_feed(
                inputs, mode="train"),
            fetch_dict=self.train_fetch_dict)

    def evaluate(self, inputs):
        return self.execute(
            program=self.eval_program,
            feed=self.set_feed(
                inputs, mode="evaluate"),
            fetch_dict=self.eval_fetch_dict)

    def infer(self, inputs):
        batch_size = inputs["size"]
        result = self.execute(
            program=self.infer_program,
            feed=self.set_feed(
                inputs, mode="infer"),
            fetch_dict=self.infer_fetch_dict,
            return_numpy=False)

        def select_top1_in_beam(T):
            lod = T.lod()
            lens = T.recursive_sequence_lengths()[-1]
            sents = np.split(np.array(T), lod[-1][1:-1])
            top1_ids = lod[0][:-1]
            data = np.concatenate([sents[i] for i in top1_ids])
            recur_lens = [[1 for _ in top1_ids], [lens[i] for i in top1_ids]]
            return fluid.create_lod_tensor(data, recur_lens, self.place)

        preds = select_top1_in_beam(result["preds"])
        lens = preds.recursive_sequence_lengths()
        lens[0] = [self.num_mappings] * batch_size
        preds.set_recursive_sequence_lengths(lens)
        result["preds"] = preds

        return result

    def set_feed(self, inputs, mode="train"):
        feed = {}
        feed["post"] = inputs["post"]
        feed["response"] = inputs["response"]
        if mode == "infer":
            start_id = self.generator.start_id
            batch_size = inputs["size"]
            batch_size = batch_size * self.num_mappings
            init_ids_data = np.array(
                [[start_id] for _ in range(batch_size)], dtype='int64')
            init_recursive_seq_lens = [[1] * batch_size, [1] * batch_size]
            init_ids = fluid.create_lod_tensor(
                init_ids_data, init_recursive_seq_lens, self.place)
            feed["init_ids"] = init_ids

            post_lens = inputs["post"].recursive_sequence_lengths()[0]
            post_expand_lens = [
                l for l in post_lens for _ in range(self.num_mappings)
            ]
            post_expand_lens.insert(0, 0)
            post_expand_lod = np.cumsum(post_expand_lens)[None, :]
            post_expand_lod = post_expand_lod.astype("int32")
            feed["post_expand_lod"] = post_expand_lod
        else:
            feed["label"] = inputs["label"]
            feed["pos_response"] = sequence_but(
                inputs["response"], self.place, position="first")
        return feed
