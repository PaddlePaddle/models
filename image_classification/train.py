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
# limitations under the License

import gzip

import paddle.v2 as paddle
import reader
import vgg

DATA_DIM = 3 * 224 * 224
CLASS_DIM = 1000
BATCH_SIZE = 128


def main():

    # PaddlePaddle init
    paddle.init(use_gpu=True, trainer_count=4)

    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(DATA_DIM))
    lbl = paddle.layer.data(
        name="label", type=paddle.data_type.integer_value(CLASS_DIM))
    net = vgg.vgg13(image)
    out = paddle.layer.fc(
        input=net, size=CLASS_DIM, act=paddle.activation.Softmax())
    cost = paddle.layer.classification_cost(input=out, label=lbl)

    # Create parameters
    parameters = paddle.parameters.create(cost)

    # Create optimizer
    optimizer = paddle.optimizer.Momentum(
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0005 *
                                                         BATCH_SIZE),
        learning_rate=0.01 / BATCH_SIZE,
        learning_rate_decay_a=0.1,
        learning_rate_decay_b=128000 * 35,
        learning_rate_schedule="discexp", )

    train_reader = paddle.batch(
        paddle.reader.shuffle(reader.test_reader("train.list"), buf_size=1000),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        reader.train_reader("test.list"), batch_size=BATCH_SIZE)

    # End batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 1 == 0:
                print "\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
        if isinstance(event, paddle.event.EndPass):
            with gzip.open('params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
                parameters.to_tar(f)

            result = trainer.test(reader=test_reader)
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

    # Create trainer
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    trainer.train(
        reader=train_reader, num_passes=200, event_handler=event_handler)


if __name__ == '__main__':
    main()
