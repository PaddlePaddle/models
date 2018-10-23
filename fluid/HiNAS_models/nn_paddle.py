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
import time

import numpy as np
import paddle
import paddle.fluid as fluid
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
flags.DEFINE_float("gd_clip", 0, "gradient clipping. 0 for disable")

flags.DEFINE_boolean("shuffle_image", True, "shuffle input images on training")

flags.DEFINE_boolean("use_nccl", False, "Parallel training")

flags.DEFINE_string(
    "save_model_path", None,
    "Save model to 'save_model_path' (directory) after training.")
flags.DEFINE_string("load_model_path", None,
                    "Load pre-trained model. (skip training)")
flags.DEFINE_string("model_filename", "hinas.model",
                    "model filename to save or load.")

dataset_train_size = 50000


class Model(object):
    def __init__(self, build_fn, tokens):
        print("learning rate: %f -> %f, cosine annealing" %
              (FLAGS.lr_max, FLAGS.lr_min))
        print("epoch: %d" % FLAGS.num_epochs)
        print("batch size: %d" % FLAGS.batch_size)
        print("L2 decay: %f" % FLAGS.weight_decay)
        print("gd clip: %f" % FLAGS.gd_clip)
        print("parallel training: %s" % FLAGS.use_nccl)

        self.max_step = dataset_train_size * FLAGS.num_epochs // FLAGS.batch_size

        self.build_fn = build_fn
        self.tokens = tokens
        print("Token is %s" % ",".join(map(str, tokens)))

        self.best_acc = 0

    def cosine_annealing(self):
        step = _decay_step_counter()
        lr = FLAGS.lr_min + (FLAGS.lr_max - FLAGS.lr_min) / 2 \
             * (1.0 + fluid.layers.ops.cos(step / self.max_step * math.pi))
        return lr

    def test(self, test_reader, prog, exe, feeder, avg_loss, accuracy):
        test_costs = []
        test_accs = []
        for data in test_reader():
            if FLAGS.use_nccl:
                cost, acc = exe.run(feed=feeder.feed(data),
                                    fetch_list=[avg_loss.name, accuracy.name])
            else:
                cost, acc = exe.run(prog,
                                    feed=feeder.feed(data),
                                    fetch_list=[avg_loss.name, accuracy.name])
            test_costs.append(cost)
            test_accs.append(acc)

        test_acc = np.mean(test_accs)
        print("Test done: Loss %f, Acc %f" % (np.mean(test_costs), test_acc))

        if test_acc > self.best_acc:
            self.best_acc = test_acc
            if FLAGS.save_model_path is not None:
                print("Model is saved to" + FLAGS.save_model_path)
                fluid.io.save_params(
                    executor=exe,
                    dirname=FLAGS.save_model_path,
                    main_program=fluid.default_main_program(),
                    filename=FLAGS.model_filename)

        print("Best acc %f" % self.best_acc)

    def run(self):
        # input data
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

        images = fluid.layers.data(
            name='pixel', shape=[3, 32, 32], dtype='float32')
        labels = fluid.layers.data(name='label', shape=[1], dtype='int64')

        # train network
        avg_loss, accuracy = self.build_fn(images, labels, self.tokens)

        # gradient clipping
        if FLAGS.gd_clip > 0:
            fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByValue(
                max=FLAGS.gd_clip, min=-FLAGS.gd_clip))

        test_program = fluid.default_main_program().clone(for_test=True)

        optimizer = fluid.optimizer.Momentum(
            learning_rate=self.cosine_annealing(),
            momentum=FLAGS.momentum,
            use_nesterov=True,
            regularization=fluid.regularizer.L2DecayRegularizer(
                FLAGS.weight_decay))
        optimizer.minimize(avg_loss)

        # run
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        if FLAGS.use_nccl:
            train_exe = fluid.ParallelExecutor(
                use_cuda=True,
                loss_name=avg_loss.name,
                main_program=fluid.default_main_program())
            test_exe = fluid.ParallelExecutor(
                use_cuda=True,
                share_vars_from=train_exe,
                main_program=test_program)

        feeder = fluid.DataFeeder(place=place, feed_list=[images, labels])

        if FLAGS.load_model_path is not None:
            print("loading pre-trainer model...")
            fluid.io.load_params(
                executor=train_exe if FLAGS.use_nccl else exe,
                dirname=FLAGS.load_model_path,
                main_program=fluid.default_main_program(),
                filename=FLAGS.model_filename)
            print("run testing...")
            self.test(test_reader, test_program, test_exe
                      if FLAGS.use_nccl else exe, feeder, avg_loss, accuracy)
            return

        costs = []
        accs = []
        for epoch in range(FLAGS.num_epochs):
            start_time = time.time()
            for batch, data in enumerate(train_reader()):
                if FLAGS.use_nccl:
                    cost, acc = train_exe.run(
                        feed=feeder.feed(data),
                        fetch_list=[avg_loss.name, accuracy.name])
                else:
                    cost, acc = exe.run(
                        fluid.default_main_program(),
                        feed=feeder.feed(data),
                        fetch_list=[avg_loss.name, accuracy.name],
                        use_program_cache=True)
                costs.append(cost)
                accs.append(acc)
                if batch % 10 == 0:
                    print("Epoch %d, Step %d, Loss %f, Acc %f" %
                          (epoch, batch, np.mean(costs), np.mean(accs)))
                    del costs[:]
                    del accs[:]
            print("Epoch done. time elapsed: {}s"
                  .format(time.time() - start_time))

            if epoch % 3 == 0 or epoch == FLAGS.num_epochs - 1:
                self.test(test_reader, test_program, test_exe if FLAGS.use_nccl
                          else exe, feeder, avg_loss, accuracy)

        print("Train model done.")
