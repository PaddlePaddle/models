# -*- encoding:utf-8 -*-
import paddle.v2 as paddle
import gzip

from nce_conf import network_conf


def main():
    paddle.init(use_gpu=False, trainer_count=1)
    word_dict = paddle.dataset.imikolov.build_dict()
    dict_size = len(word_dict)

    cost = network_conf(
        is_train=True, hidden_size=256, embedding_size=32, dict_size=dict_size)

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

            model_name = "./models/model_pass_%05d.tar.gz" % event.pass_id
            print "Save model into %s ..." % model_name
            with gzip.open(model_name, 'w') as f:
                parameters.to_tar(f)

    feeding = {
        'firstw': 0,
        'secondw': 1,
        'thirdw': 2,
        'fourthw': 3,
        'fifthw': 4
    }

    trainer.train(
        paddle.batch(paddle.dataset.imikolov.train(word_dict, 5)(), 64),
        num_passes=1000,
        event_handler=event_handler,
        feeding=feeding)


if __name__ == '__main__':
    main()
