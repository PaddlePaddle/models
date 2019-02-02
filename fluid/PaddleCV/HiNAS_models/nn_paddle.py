# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
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

import math

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.trainer import *
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
import reader

from absl import flags

# import preprocess

FLAGS = flags.FLAGS

flags.DEFINE_float("lr_max", 0.1, "initial learning rate")
flags.DEFINE_float("lr_min", 0.0001, "limiting learning rate")

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("num_epochs", 200, "total epochs to train")
flags.DEFINE_float("weight_decay", 0.0004, "weight decay")

flags.DEFINE_float("momentum", 0.9, "momentum")

flags.DEFINE_boolean("shuffle_image", True, "shuffle input images on training")

dataset_train_size = 50000


class Model(object):
    def __init__(self, build_fn, tokens):
        print("learning rate: %f -> %f, cosine annealing" %
              (FLAGS.lr_max, FLAGS.lr_min))
        print("epoch: %d" % FLAGS.num_epochs)
        print("batch size: %d" % FLAGS.batch_size)
        print("L2 decay: %f" % FLAGS.weight_decay)

        self.max_step = dataset_train_size * FLAGS.num_epochs // FLAGS.batch_size

        self.build_fn = build_fn
        self.tokens = tokens
        print("Token is %s" % ",".join(map(str, tokens)))

    def cosine_annealing(self):
        step = _decay_step_counter()
        lr = FLAGS.lr_min + (FLAGS.lr_max - FLAGS.lr_min) / 2 \
             * (1.0 + fluid.layers.ops.cos(step / self.max_step * math.pi))
        return lr

    def optimizer_program(self):
        return fluid.optimizer.Momentum(
            learning_rate=self.cosine_annealing(),
            momentum=FLAGS.momentum,
            use_nesterov=True,
            regularization=fluid.regularizer.L2DecayRegularizer(
                FLAGS.weight_decay))

    def inference_network(self):
        images = fluid.layers.data(
            name='pixel', shape=[3, 32, 32], dtype='float32')
        return self.build_fn(images, self.tokens)

    def train_network(self):
        predict = self.inference_network()
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(cost)
        accuracy = fluid.layers.accuracy(input=predict, label=label)
        # self.parameters = fluid.parameters.create(avg_cost)
        return [avg_cost, accuracy]

    def run(self):
        train_files = reader.train10()
        test_files = reader.test10()

        if FLAGS.shuffle_image:
            train_reader = paddle.batch(
                paddle.reader.shuffle(train_files, dataset_train_size),
                batch_size=FLAGS.batch_size)
        else:
            train_reader = paddle.batch(
                train_files, batch_size=FLAGS.batch_size)

        test_reader = paddle.batch(test_files, batch_size=FLAGS.batch_size)

        costs = []
        accs = []

        def event_handler(event):
            if isinstance(event, EndStepEvent):
                costs.append(event.metrics[0])
                accs.append(event.metrics[1])
                if event.step % 20 == 0:
                    print("Epoch %d, Step %d, Loss %f, Acc %f" % (
                        event.epoch, event.step, np.mean(costs), np.mean(accs)))
                    del costs[:]
                    del accs[:]

            if isinstance(event, EndEpochEvent):
                if event.epoch % 3 == 0 or event.epoch == FLAGS.num_epochs - 1:
                    avg_cost, accuracy = trainer.test(
                        reader=test_reader, feed_order=['pixel', 'label'])

                    event_handler.best_acc = max(event_handler.best_acc,
                                                 accuracy)
                    print("Test with epoch %d, Loss %f, Acc %f" %
                          (event.epoch, avg_cost, accuracy))
                    print("Best acc %f" % event_handler.best_acc)

        event_handler.best_acc = 0.0
        place = fluid.CUDAPlace(0)
        trainer = Trainer(
            train_func=self.train_network,
            optimizer_func=self.optimizer_program,
            place=place)

        trainer.train(
            reader=train_reader,
            num_epochs=FLAGS.num_epochs,
            event_handler=event_handler,
            feed_order=['pixel', 'label'])
