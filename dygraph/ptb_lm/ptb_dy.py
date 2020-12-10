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

from __future__ import print_function

import os
import unittest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.dygraph.nn import Embedding
import paddle.fluid.framework as framework
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.dygraph.base import to_variable
import numpy as np
import six
import multiprocessing

import reader
import model_check
import time

from args import *

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")


class TimeCostAverage(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.cnt = 0
        self.total_time = 0

    def record(self, usetime):
        self.cnt += 1
        self.total_time += usetime

    def get_average(self):
        if self.cnt == 0:
            return 0
        return self.total_time / self.cnt


class SimpleLSTMRNN(fluid.Layer):
    def __init__(self,
                 hidden_size,
                 num_steps,
                 num_layers=2,
                 init_scale=0.1,
                 dropout=None):
        super(SimpleLSTMRNN, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._init_scale = init_scale
        self._dropout = dropout
        self._num_steps = num_steps
        self.cell_array = []
        self.hidden_array = []

        self.weight_1_arr = []
        self.weight_2_arr = []
        self.bias_arr = []
        self.mask_array = []

        for i in range(self._num_layers):
            weight_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 2, self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_1_arr.append(self.add_parameter('w_%d' % i, weight_1))
            bias_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.Constant(0.0))
            self.bias_arr.append(self.add_parameter('b_%d' % i, bias_1))

    def forward(self, input_embedding, init_hidden=None, init_cell=None):
        cell_array = []
        hidden_array = []

        for i in range(self._num_layers):
            hidden_array.append(init_hidden[i])
            cell_array.append(init_cell[i])

        res = []
        for index in range(self._num_steps):
            step_input = input_embedding[:, index, :]
            for k in range(self._num_layers):
                pre_hidden = hidden_array[k]
                pre_cell = cell_array[k]
                weight_1 = self.weight_1_arr[k]
                bias = self.bias_arr[k]

                nn = fluid.layers.concat([step_input, pre_hidden], 1)
                gate_input = fluid.layers.matmul(x=nn, y=weight_1)

                gate_input = fluid.layers.elementwise_add(gate_input, bias)
                i, j, f, o = fluid.layers.split(
                    gate_input, num_or_sections=4, dim=-1)
                c = pre_cell * fluid.layers.sigmoid(f) + fluid.layers.sigmoid(
                    i) * fluid.layers.tanh(j)
                m = fluid.layers.tanh(c) * fluid.layers.sigmoid(o)
                hidden_array[k] = m
                cell_array[k] = c
                step_input = m

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
        last_cell = fluid.layers.concat(cell_array, 1)
        last_cell = fluid.layers.reshape(
            last_cell, shape=[-1, self._num_layers, self._hidden_size])
        last_cell = fluid.layers.transpose(x=last_cell, perm=[1, 0, 2])
        return real_res, last_hidden, last_cell


class PtbModel(fluid.Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 num_layers=2,
                 num_steps=20,
                 init_scale=0.1,
                 dropout=None):
        super(PtbModel, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_scale = init_scale
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.dropout = dropout
        self.simple_lstm_rnn = SimpleLSTMRNN(
            hidden_size,
            num_steps,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=dropout)
        self.embedding = Embedding(
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

    def build_once(self, input, label, init_hidden, init_cell):
        pass

    def forward(self, input, label, init_hidden, init_cell):

        init_h = fluid.layers.reshape(
            init_hidden, shape=[self.num_layers, -1, self.hidden_size])

        init_c = fluid.layers.reshape(
            init_cell, shape=[self.num_layers, -1, self.hidden_size])

        x_emb = self.embedding(input)

        x_emb = fluid.layers.reshape(
            x_emb, shape=[-1, self.num_steps, self.hidden_size])
        if self.dropout is not None and self.dropout > 0.0:
            x_emb = fluid.layers.dropout(
                x_emb,
                dropout_prob=self.dropout,
                dropout_implementation='upscale_in_train')
        rnn_out, last_hidden, last_cell = self.simple_lstm_rnn(x_emb, init_h,
                                                               init_c)

        projection = fluid.layers.matmul(rnn_out, self.softmax_weight)
        projection = fluid.layers.elementwise_add(projection, self.softmax_bias)

        loss = fluid.layers.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False)
        loss = fluid.layers.reshape(loss, shape=[-1, self.num_steps])
        loss = fluid.layers.reduce_mean(loss, dim=[0])
        loss = fluid.layers.reduce_sum(loss)

        return loss, last_hidden, last_cell

    def debug_emb(self):

        np.save("emb_grad", self.x_emb.gradient())


def train_ptb_lm():
    args = parse_args()

    # check if set use_gpu=True in paddlepaddle cpu version
    model_check.check_cuda(args.use_gpu)

    place = core.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    # check if paddlepaddle version is satisfied
    model_check.check_version()

    model_type = args.model_type

    vocab_size = 10000
    if model_type == "test":
        num_layers = 1
        batch_size = 2
        hidden_size = 10
        num_steps = 3
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
        max_epoch = 13
        dropout = 0.0
        lr_decay = 0.5
        base_learning_rate = 1.0
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

    with fluid.dygraph.guard(place):
        if args.ce:
            print("ce mode")
            seed = 33
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            max_epoch = 1
        ptb_model = PtbModel(
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
        last_cell = None

        data_path = args.data_path
        print("begin to load data")
        ptb_data = reader.get_ptb_data(data_path)
        print("finished load data")
        train_data, valid_data, test_data = ptb_data

        batch_len = len(train_data) // batch_size
        total_batch_size = (batch_len - 1) // num_steps
        log_interval = 200

        bd = []
        lr_arr = [1.0]
        for i in range(1, max_epoch):
            bd.append(total_batch_size * i)
            new_lr = base_learning_rate * (lr_decay**
                                           max(i + 1 - epoch_start_decay, 0.0))
            lr_arr.append(new_lr)

        grad_clip = fluid.clip.GradientClipByGlobalNorm(max_grad_norm)
        sgd = SGDOptimizer(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr_arr),
            parameter_list=ptb_model.parameters(),
            grad_clip=grad_clip)

        def reader_decorator(reader):
            def __reader__():
                for item in reader:
                    x_data = item[0].reshape((-1, num_steps, 1))
                    y_data = item[1].reshape((-1, num_steps, 1))
                    yield x_data, y_data

            return __reader__

        def eval(model, data):
            print("begin to eval")
            total_loss = 0.0
            iters = 0.0
            init_hidden_data = np.zeros(
                (num_layers, batch_size, hidden_size), dtype='float32')
            init_cell_data = np.zeros(
                (num_layers, batch_size, hidden_size), dtype='float32')

            model.eval()
            train_data_iter = reader_decorator(
                reader.get_data_iter(data, batch_size, num_steps))

            eval_data_loader = fluid.io.DataLoader.from_generator(capacity=200)
            eval_data_loader.set_batch_generator(train_data_iter, places=place)

            for batch_id, batch in enumerate(eval_data_loader):
                x, y = batch
                init_hidden = to_variable(init_hidden_data)
                init_cell = to_variable(init_cell_data)
                dy_loss, last_hidden, last_cell = ptb_model(x, y, init_hidden,
                                                            init_cell)

                out_loss = dy_loss.numpy()

                init_hidden_data = last_hidden.numpy()
                init_cell_data = last_cell.numpy()

                total_loss += out_loss
                iters += num_steps

            print("eval finished")
            ppl = np.exp(total_loss / iters)
            print("ppl ", batch_id, ppl[0])

        ce_time = []
        ce_ppl = []

        total_batch_num = 0  #this is for benchmark
        for epoch_id in range(max_epoch):
            epoch_start = time.time()

            ptb_model.train()
            total_loss = 0.0
            iters = 0.0
            init_hidden_data = np.zeros(
                (num_layers, batch_size, hidden_size), dtype='float32')
            init_cell_data = np.zeros(
                (num_layers, batch_size, hidden_size), dtype='float32')

            train_data_iter = reader_decorator(
                reader.get_data_iter(train_data, batch_size, num_steps))

            train_data_loader = fluid.io.DataLoader.from_generator(capacity=200)
            train_data_loader.set_batch_generator(train_data_iter, places=place)

            init_hidden = to_variable(init_hidden_data)
            init_cell = to_variable(init_cell_data)

            batch_cost_avg = TimeCostAverage()
            reader_cost_avg = TimeCostAverage()

            batch_start = time.time()
            for batch_id, batch in enumerate(train_data_loader):
                if args.max_iter and total_batch_num == args.max_iter:
                    return

                train_reader_cost = time.time() - batch_start
                reader_cost_avg.record(train_reader_cost)

                x, y = batch

                dy_loss, last_hidden, last_cell = ptb_model(x, y, init_hidden,
                                                            init_cell)
                init_hidden = last_hidden.detach()
                init_cell = last_cell.detach()
                out_loss = dy_loss.numpy()

                dy_loss.backward()
                sgd.minimize(dy_loss)
                ptb_model.clear_gradients()

                global_lr = sgd._global_learning_rate().numpy()
                total_loss += out_loss
                iters += num_steps
                total_batch_num = total_batch_num + 1  #this is for benchmark

                train_batch_cost = time.time() - batch_start
                batch_cost_avg.record(train_batch_cost)

                if batch_id > 0 and batch_id % log_interval == 0:
                    ppl = np.exp(total_loss / iters)
                    print(
                        "-- Epoch:[%d]; Batch:[%d]; ppl: %.5f, lr: %.5f, loss: %.5f, batch_cost: %.5f sec, reader_cost: %.5f sec, ips: %.5f words/sec"
                        % (epoch_id, batch_id, ppl[0], global_lr, out_loss,
                           batch_cost_avg.get_average(),
                           reader_cost_avg.get_average(),
                           batch_size / batch_cost_avg.get_average()))
                    batch_cost_avg.reset()
                    reader_cost_avg.reset()
                batch_start = time.time()

            ppl = np.exp(total_loss / iters)
            train_epoch_cost = time.time() - epoch_start
            print("-- Epoch:[%d]; ppl: %.5f, epoch_cost: %.5f s" %
                  (epoch_id, ppl[0], train_epoch_cost))

            ce_time.append(train_epoch_cost)
            ce_ppl.append(ppl[0])

            if batch_size <= 20 and epoch_id == 0 and ppl[0] > 1000:
                # for bad init, after first epoch, the loss is over 1000
                # no more need to continue
                print(
                    "Parameters are randomly initialized and not good this time because the loss is over 1000 after the first epoch."
                )
                print("Abort this training process and please start again.")
                return

            save_model_dir = os.path.join(args.save_model_dir,
                                          str(epoch_id), 'params')
            fluid.save_dygraph(ptb_model.state_dict(), save_model_dir)
            print("Saved model to: %s.\n" % save_model_dir)

            eval(ptb_model, valid_data)

        if args.ce:
            _ppl = 0
            _time = 0
            try:
                _time = ce_time[-1]
                _ppl = ce_ppl[-1]
            except:
                print("ce info error")
            print("kpis\ttrain_duration_card%s\t%s" % (dev_count, _time))
            print("kpis\ttrain_ppl_card%s\t%f" % (dev_count, _ppl))

        eval(ptb_model, test_data)


train_ptb_lm()
