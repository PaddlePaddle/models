import sys
import sqlite3
import paddle.v2 as paddle
import ptb
import gzip


def rnn_net(dict_dim, is_generating=False):
    # dimension of word vector
    word_vector_dim = 1024
    #dimension of hidden unit in GRU Encoder network
    encoder_size = 1024

    # Encoder
    src_word_id = paddle.layer.data(
        name='source_word',
        type=paddle.data_type.integer_value_sequence(dict_dim))
    target_word_id = paddle.layer.data(
        name='target_word',
        type=paddle.data_type.integer_value_sequence(dict_dim))

    src_embedding = paddle.layer.embedding(
        input=src_word_id,
        size=word_vector_dim,
        param_attr=paddle.attr.ParamAttr(name='_source_word_embedding'))

    src_forward = paddle.networks.simple_gru(
        input=src_embedding, size=encoder_size)
    src_backward = paddle.networks.simple_gru(
        input=src_embedding, size=encoder_size, reverse=True)
    encoded_vector = paddle.layer.concat(input=[src_forward, src_backward])
    output = paddle.layer.fc(
        input=encoded_vector,
        size=dict_dim,
        act=paddle.activation.Softmax(),
        param_attr=paddle.attr.Param(initial_std=0.01))
    # calc cost
    cost = paddle.layer.classification_cost(input=output, label=target_word_id)

    return cost


def main():
    paddle.init(use_gpu=False, trainer_count=8)
    is_generating = False

    # source and target dict dim.
    word_dict = ptb.build_dict()
    dict_size = len(word_dict)
    # train the network
    if not is_generating:
        cost = rnn_net(dict_size)
        parameters = paddle.parameters.create(cost)

        # define optimize method and trainer
        optimizer = paddle.optimizer.Adam(
            learning_rate=5e-3,
            regularization=paddle.optimizer.L2Regularization(rate=8e-4))

        trainer = paddle.trainer.SGD(
            cost=cost, parameters=parameters, update_equation=optimizer)

        # define data reader
        ptb_reader = paddle.batch(
            paddle.reader.shuffle(ptb.seq_train(word_dict, 100), buf_size=8192),
            batch_size=64)

        # define event_handler callback
        def event_handler(event):
            # save model
            if isinstance(event, paddle.event.EndPass):
                # need make directory models in current dir before running
                model_name = './models/model_pass_%05d.tar.gz' % event.pass_id
                print("Save model into %s ..." % model_name)
                with gzip.open(model_name, 'w') as f:
                    parameters.to_tar(f)

            # output training info
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 10 == 0:
                    result = trainer.test(
                        paddle.batch(
                            paddle.reader.shuffle(
                                ptb.seq_test(word_dict, 100), buf_size=8192),
                            batch_size=64))
                    print "\nPass %d, Batch %d, Cost %f, %s, Test Cost %f, %s"\
                        % (
                            event.pass_id, event.batch_id, event.cost,
                            event.metrics, result.cost, result.metrics)
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()

        # start to train
        feeding = {'source_word': 0, 'target_word': 1}
        trainer.train(
            reader=ptb_reader,
            event_handler=event_handler,
            num_passes=2,
            feeding=feeding)


if __name__ == '__main__':
    main()
