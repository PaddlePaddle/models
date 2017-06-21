#!/usr/bin/env python

from network_conf import *


def train(source_dict_dim, target_dict_dim):
    '''
    Training function for NMT

    :param source_dict_dim: size of source dictionary
    :type source_dict_dim: int
    :param target_dict_dim: size of target dictionary
    :type target_dict_dim: int
    '''
    # initialize model
    paddle.init(use_gpu=False, trainer_count=1)

    cost = seq2seq_net(source_dict_dim, target_dict_dim)
    parameters = paddle.parameters.create(cost)

    # define optimize method and trainer
    optimizer = paddle.optimizer.RMSProp(
        learning_rate=1e-3,
        gradient_clipping_threshold=10.0,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4))
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)
    # define data reader
    wmt14_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(source_dict_dim), buf_size=8192),
        batch_size=8)

    # define event_handler callback
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if not event.batch_id % 500 and event.batch_id:
                with gzip.open("models/nmt_without_att_params_batch_%05d.tar.gz"
                               % event.batch_id, "w") as f:
                    parameters.to_tar(f)

            if event.batch_id and not event.batch_id % 10:
                print("\nPass %d, Batch %d, Cost %f, %s" %
                      (event.pass_id, event.batch_id, event.cost,
                       event.metrics))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

    # start to train
    trainer.train(
        reader=wmt14_reader, event_handler=event_handler, num_passes=2)


if __name__ == '__main__':
    train(source_dict_dim=3000, target_dict_dim=3000)
