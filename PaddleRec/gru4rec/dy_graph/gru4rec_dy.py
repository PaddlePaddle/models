#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import os
import unittest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.dygraph.nn import Embedding
import paddle.fluid.framework as framework
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.optimizer import AdagradOptimizer
from paddle.fluid.dygraph.base import to_variable
import numpy as np
import six

import reader
import model_check
import time

from args import *

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")


class SimpleGRURNN(fluid.Layer):
    def __init__(self,
                 hidden_size,
                 num_steps,
                 num_layers=2,
                 init_scale=0.1,
                 dropout=None):
        super(SimpleGRURNN, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._init_scale = init_scale
        self._dropout = dropout
        self._num_steps = num_steps

        self.weight_1_arr = []
        self.weight_2_arr = []
        self.weight_3_arr = []
        self.bias_1_arr = []
        self.bias_2_arr = []
        self.mask_array = []

        for i in range(self._num_layers):
            weight_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 2, self._hidden_size * 2],
                dtype="float32",
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_1_arr.append(self.add_parameter('w1_%d' % i, weight_1))
            weight_2 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size, self._hidden_size],
                dtype="float32",
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_2_arr.append(self.add_parameter('w2_%d' % i, weight_2))
            weight_3 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size, self._hidden_size],
                dtype="float32",
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_3_arr.append(self.add_parameter('w3_%d' % i, weight_3))
            bias_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 2],
                dtype="float32",
                default_initializer=fluid.initializer.Constant(0.0))
            self.bias_1_arr.append(self.add_parameter('b1_%d' % i, bias_1))
            bias_2 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 1],
                dtype="float32",
                default_initializer=fluid.initializer.Constant(0.0))
            self.bias_2_arr.append(self.add_parameter('b2_%d' % i, bias_2))

    def forward(self, input_embedding, init_hidden=None):
        hidden_array = []

        for i in range(self._num_layers):
            hidden_array.append(init_hidden[i])

        res = []
        for index in range(self._num_steps):
            step_input = input_embedding[:, index, :]
            for k in range(self._num_layers):
                pre_hidden = hidden_array[k]
                weight_1 = self.weight_1_arr[k]
                weight_2 = self.weight_2_arr[k]
                weight_3 = self.weight_3_arr[k]
                bias_1 = self.bias_1_arr[k]
                bias_2 = self.bias_2_arr[k]

                nn = fluid.layers.concat([step_input, pre_hidden], 1)
                gate_input = fluid.layers.matmul(x=nn, y=weight_1)
                gate_input = fluid.layers.elementwise_add(gate_input, bias_1)
                u, r = fluid.layers.split(gate_input, num_or_sections=2, dim=-1)
                hidden_c = fluid.layers.tanh(
                    fluid.layers.elementwise_add(
                        fluid.layers.matmul(
                            x=step_input, y=weight_2) + fluid.layers.matmul(
                                x=(fluid.layers.sigmoid(r) * pre_hidden),
                                y=weight_3),
                        bias_2))
                hidden_state = fluid.layers.sigmoid(u) * pre_hidden + (
                    1.0 - fluid.layers.sigmoid(u)) * hidden_c
                hidden_array[k] = hidden_state
                step_input = hidden_state

                if self._dropout is not None and self._dropout > 0.0:
                    step_input = fluid.layers.dropout(
                        step_input,
                        dropout_prob=self._dropout,
                        dropout_implementation='upscale_in_train')
            res.append(step_input)
        real_res = fluid.layers.concat(res, 1)
        real_res = fluid.layers.reshape(
            real_res, [-1, self._num_steps, self._hidden_size])
        last_hidden = fluid.layers.concat(hidden_array, 1)
        last_hidden = fluid.layers.reshape(
            last_hidden, shape=[-1, self._num_layers, self._hidden_size])
        last_hidden = fluid.layers.transpose(x=last_hidden, perm=[1, 0, 2])
        return real_res, last_hidden


class PtbModel(fluid.Layer):
    def __init__(self,
                 name_scope,
                 hidden_size,
                 vocab_size,
                 num_layers=2,
                 num_steps=20,
                 init_scale=0.1,
                 dropout=None):
        #super(PtbModel, self).__init__(name_scope)
        super(PtbModel, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_scale = init_scale
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.dropout = dropout
        self.simple_gru_rnn = SimpleGRURNN(
            #self.full_name(),
            hidden_size,
            num_steps,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=dropout)
        self.embedding = Embedding(
            #self.full_name(),
            size=[vocab_size, hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale)))
        self.softmax_weight = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.hidden_size, self.vocab_size],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))
        self.softmax_bias = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.vocab_size],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

    def build_once(self, input, label, init_hidden):
        pass

    def forward(self, input, label, init_hidden):

        init_h = fluid.layers.reshape(
            init_hidden, shape=[self.num_layers, -1, self.hidden_size])

        x_emb = self.embedding(input)

        x_emb = fluid.layers.reshape(
            x_emb, shape=[-1, self.num_steps, self.hidden_size])
        if self.dropout is not None and self.dropout > 0.0:
            x_emb = fluid.layers.dropout(
                x_emb,
                dropout_prob=self.dropout,
                dropout_implementation='upscale_in_train')
        rnn_out, last_hidden = self.simple_gru_rnn(x_emb, init_h)

        projection = fluid.layers.matmul(rnn_out, self.softmax_weight)
        projection = fluid.layers.elementwise_add(projection, self.softmax_bias)
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False)
        pre_2d = fluid.layers.reshape(projection, shape=[-1, self.vocab_size])
        label_2d = fluid.layers.reshape(label, shape=[-1, 1])
        acc = fluid.layers.accuracy(input=pre_2d, label=label_2d, k=20)
        loss = fluid.layers.reshape(loss, shape=[-1, self.num_steps])
        loss = fluid.layers.reduce_mean(loss, dim=[0])
        loss = fluid.layers.reduce_sum(loss)

        return loss, last_hidden, acc

    def debug_emb(self):

        np.save("emb_grad", self.x_emb.gradient())


def train_ptb_lm():
    args = parse_args()

    # check if set use_gpu=True in paddlepaddle cpu version
    model_check.check_cuda(args.use_gpu)
    # check if paddlepaddle version is satisfied
    model_check.check_version()

    model_type = args.model_type

    vocab_size = 37484
    if model_type == "test":
        num_layers = 1
        batch_size = 2
        hidden_size = 10
        num_steps = 4
        init_scale = 0.1
        max_grad_norm = 5.0
        epoch_start_decay = 1
        max_epoch = 1
        dropout = 0.0
        lr_decay = 0.5
        base_learning_rate = 1.0
    elif model_type == "small":
        num_layers = 2
        batch_size = 20
        hidden_size = 200
        num_steps = 20
        init_scale = 0.1
        max_grad_norm = 5.0
        epoch_start_decay = 4
        max_epoch = 2
        dropout = 0.0
        lr_decay = 0.5
        base_learning_rate = 1.0
    elif model_type == "gru4rec":
        num_layers = 1
        batch_size = 500
        hidden_size = 100
        num_steps = 10
        init_scale = 0.1
        max_grad_norm = 5.0
        epoch_start_decay = 10
        max_epoch = 5
        dropout = 0.0
        lr_decay = 0.5
        base_learning_rate = 0.05
    elif model_type == "medium":
        num_layers = 2
        batch_size = 20
        hidden_size = 650
        num_steps = 35
        init_scale = 0.05
        max_grad_norm = 5.0
        epoch_start_decay = 6
        max_epoch = 39
        dropout = 0.5
        lr_decay = 0.8
        base_learning_rate = 1.0
    elif model_type == "large":
        num_layers = 2
        batch_size = 20
        hidden_size = 1500
        num_steps = 35
        init_scale = 0.04
        max_grad_norm = 10.0
        epoch_start_decay = 14
        max_epoch = 55
        dropout = 0.65
        lr_decay = 1.0 / 1.15
        base_learning_rate = 1.0
    else:
        print("model type not support")
        return

    with fluid.dygraph.guard(core.CUDAPlace(0)):
        if args.ce:
            print("ce mode")
            seed = 33
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            max_epoch = 1
        ptb_model = PtbModel(
            "ptb_model",
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_steps=num_steps,
            init_scale=init_scale,
            dropout=dropout)

        if args.init_from_pretrain_model:
            if not os.path.exists(args.init_from_pretrain_model + '.pdparams'):
                print(args.init_from_pretrain_model)
                raise Warning("The pretrained params do not exist.")
                return
            fluid.load_dygraph(args.init_from_pretrain_model)
            print("finish initing model from pretrained params from %s" %
                  (args.init_from_pretrain_model))

        dy_param_updated = dict()
        dy_param_init = dict()
        dy_loss = None
        last_hidden = None

        data_path = args.data_path
        print("begin to load data")
        ptb_data = reader.get_ptb_data(data_path)
        print("finished load data")
        train_data, valid_data, test_data = ptb_data

        batch_len = len(train_data) // batch_size
        total_batch_size = (batch_len - 1) // num_steps
        print("total_batch_size:", total_batch_size)
        log_interval = total_batch_size // 20

        bd = []
        lr_arr = [base_learning_rate]
        for i in range(1, max_epoch):
            bd.append(total_batch_size * i)
            new_lr = base_learning_rate * (lr_decay**
                                           max(i + 1 - epoch_start_decay, 0.0))
            lr_arr.append(new_lr)

        sgd = AdagradOptimizer(
            parameter_list=ptb_model.parameters(),
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr_arr))

        print("parameters:--------------------------------")
        for para in ptb_model.parameters():
            print(para.name)
        print("parameters:--------------------------------")

        def eval(model, data):
            print("begion to eval")
            total_loss = 0.0
            iters = 0.0
            init_hidden_data = np.zeros(
                (num_layers, batch_size, hidden_size), dtype='float32')

            model.eval()
            train_data_iter = reader.get_data_iter(data, batch_size, num_steps)
            init_hidden = to_variable(init_hidden_data)
            accum_num_recall = 0.0
            for batch_id, batch in enumerate(train_data_iter):
                x_data, y_data = batch
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, num_steps, 1))
                x = to_variable(x_data)
                y = to_variable(y_data)
                dy_loss, last_hidden, acc = ptb_model(x, y, init_hidden)

                out_loss = dy_loss.numpy()
                acc_ = acc.numpy()[0]
                accum_num_recall += acc_
                if batch_id % 1 == 0:
                    print("batch_id:%d  recall@20:%.4f" %
                          (batch_id, accum_num_recall / (batch_id + 1)))

                init_hidden = last_hidden

                total_loss += out_loss
                iters += num_steps

            print("eval finished")
            ppl = np.exp(total_loss / iters)
            print("recall@20 ", accum_num_recall / (batch_id + 1))
            if args.ce:
                print("kpis\ttest_ppl\t%0.3f" % ppl[0])

        grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(max_grad_norm)
        for epoch_id in range(max_epoch):
            ptb_model.train()
            total_loss = 0.0
            iters = 0.0
            init_hidden_data = np.zeros(
                (num_layers, batch_size, hidden_size), dtype='float32')

            train_data_iter = reader.get_data_iter(train_data, batch_size,
                                                   num_steps)
            init_hidden = to_variable(init_hidden_data)

            start_time = time.time()
            for batch_id, batch in enumerate(train_data_iter):
                x_data, y_data = batch
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, num_steps, 1))
                x = to_variable(x_data)
                y = to_variable(y_data)
                dy_loss, last_hidden, acc = ptb_model(x, y, init_hidden)

                out_loss = dy_loss.numpy()
                acc_ = acc.numpy()[0]

                init_hidden = last_hidden
                dy_loss.backward()
                sgd.minimize(dy_loss, grad_clip=grad_clip)
                ptb_model.clear_gradients()
                total_loss += out_loss
                iters += num_steps

                if batch_id > 0 and batch_id % 100 == 1:
                    ppl = np.exp(total_loss / iters)
                    print(
                        "-- Epoch:[%d]; Batch:[%d]; ppl: %.5f, acc: %.5f, lr: %.5f"
                        % (epoch_id, batch_id, ppl[0], acc_,
                           sgd._global_learning_rate().numpy()))

            print("one ecpoh finished", epoch_id)
            print("time cost ", time.time() - start_time)
            ppl = np.exp(total_loss / iters)
            print("-- Epoch:[%d]; ppl: %.5f" % (epoch_id, ppl[0]))
            if args.ce:
                print("kpis\ttrain_ppl\t%0.3f" % ppl[0])
            save_model_dir = os.path.join(args.save_model_dir,
                                          str(epoch_id), 'params')
            fluid.save_dygraph(ptb_model.state_dict(), save_model_dir)
            print("Saved model to: %s.\n" % save_model_dir)
            eval(ptb_model, test_data)

        #eval(ptb_model, test_data)


train_ptb_lm()
