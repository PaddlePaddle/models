#!/usr/bin/env python
#coding:gbk

from grid_lstm_net import *


def train():
    cost = grid_lstm_net(source_language_dict_dim, target_language_dict_dim)
    parameters = paddle.parameters.create(cost)
    # optimizer
    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        gradient_clipping_threshold=10.0, )

    #trainer 
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)
    #data reader 
    wmt14_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(source_dict_dim), buf_size=8192),
        batch_size=22)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0 and event.batch_id > 0:
                with gzip.open('params_grid_lstm_batch_%d.tar.gz' %
                               event.batch_id, 'w') as f:
                    parameters.to_tar(f)

            if event.batch_id % 10 == 0:
                print "\n Pass %d, Batch %d, Cost %d " % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

    trainer.train(reader=wmt14_reader, event_handler=event_handler, num_pass=2)


if __name__ == '__main__':
    train()
