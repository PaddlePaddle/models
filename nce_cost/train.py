#!/usr/bin/env python
# -*- encoding:utf-8 -*-
# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
import paddle.v2 as paddle
import gzip

from nce_conf import network_conf


def main():
    paddle.init(use_gpu=False, trainer_count=3)
    word_dict = paddle.dataset.imikolov.build_dict()
    dict_size = len(word_dict)

    cost = network_conf(
        is_train=True, hidden_size=256, embed_size=32, dict_size=dict_size)

    parameters = paddle.parameters.create(cost)
    adagrad = paddle.optimizer.AdaGrad(
        learning_rate=1e-3,
        regularization=paddle.optimizer.L2Regularization(8e-4))
    trainer = paddle.trainer.SGD(cost, parameters, adagrad)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 1000 == 0:
                print "Pass %d, Batch %d, Cost %f" % (
                    event.pass_id, event.batch_id, event.cost)

        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(
                paddle.batch(paddle.dataset.imikolov.test(word_dict, 5), 32))
            print "Test here.. Pass %d, Cost %f" % (event.pass_id, result.cost)
            with gzip.open("./model_params.tar.gz", 'w') as f:
                parameters.to_tar(f)

    feeding = {
        'firstw': 0,
        'secondw': 1,
        'thirdw': 2,
        'fourthw': 3,
        'fifthw': 4
    }

    trainer.train(
        paddle.batch(
            paddle.reader.shuffle(
                lambda: paddle.dataset.imikolov.train(word_dict, 5)(),
                buf_size=1000), 64),
        num_passes=10,
        event_handler=event_handler,
        feeding=feeding)


if __name__ == '__main__':
    main()
