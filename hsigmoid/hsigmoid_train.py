#!/usr/bin/env python
# -*- coding: utf-8 -*-

import paddle.v2 as paddle
from hsigmoid_conf import network_conf
import gzip


def main():
    paddle.init(use_gpu=False, trainer_count=1)
    word_dict = paddle.dataset.imikolov.build_dict(min_word_freq=2)
    dict_size = len(word_dict)
    cost = network_conf(
        is_train=True, hidden_size=256, embed_size=32, dict_size=dict_size)

    def event_handler(event):
        if isinstance(event, paddle.event.EndPass):
            model_name = './models/model_pass_%05d.tar.gz' % event.pass_id
            print("Save model into %s ..." % model_name)
            with gzip.open(model_name, 'w') as f:
                parameters.to_tar(f)

        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                result = trainer.test(
                    paddle.batch(
                        paddle.dataset.imikolov.test(word_dict, 5), 32))
                print("Pass %d, Batch %d, Cost %f, Test Cost %f" %
                      (event.pass_id, event.batch_id, event.cost, result.cost))

    feeding = {
        'firstw': 0,
        'secondw': 1,
        'thirdw': 2,
        'fourthw': 3,
        'fifthw': 4
    }

    parameters = paddle.parameters.create(cost)
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=3e-3,
        regularization=paddle.optimizer.L2Regularization(8e-4))
    trainer = paddle.trainer.SGD(cost, parameters, adam_optimizer)

    trainer.train(
        paddle.batch(
            paddle.reader.shuffle(
                lambda: paddle.dataset.imikolov.train(word_dict, 5)(),
                buf_size=1000), 64),
        num_passes=30,
        event_handler=event_handler,
        feeding=feeding)


if __name__ == '__main__':
    main()
