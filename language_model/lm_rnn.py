# coding=utf-8
import sys
import paddle.v2 as paddle
import data_util as reader
import gzip
import generate_text as generator

def lm(vocab_size, emb_dim, rnn_type, hidden_size, num_layer):
    """
    rnn language model definition.

    :param vocab_size: size of vocab.
    :param emb_dim: embedding vector's dimension.
    :param rnn_type: the type of RNN cell.
    :param hidden_size: number of unit.
    :param num_layer: layer number.
    :return: cost and output layer of model.
    """

    assert emb_dim > 0 and hidden_size > 0 and vocab_size > 0 and num_layer > 0

    # input layers
    data = paddle.layer.data(
        name="word", type=paddle.data_type.integer_value_sequence(vocab_size))
    target = paddle.layer.data("label", paddle.data_type.integer_value_sequence(vocab_size))

    # embedding layer
    emb = paddle.layer.embedding(input=data, size=emb_dim)

    # rnn layer
    if rnn_type == 'lstm':
        rnn_cell = paddle.networks.simple_lstm(
            input=emb, size=hidden_size)
        for _ in range(num_layer - 1):
            rnn_cell = paddle.networks.simple_lstm(
                input=rnn_cell, size=hidden_size)
    elif rnn_type == 'gru':
        rnn_cell = paddle.networks.simple_gru(
            input=emb, size=hidden_size)
        for _ in range(num_layer - 1):
            rnn_cell = paddle.networks.simple_gru(
                input=rnn_cell, size=hidden_size)
    else:
        raise Exception('rnn_type error!')

    # fc(full connected) and output layer
    output = paddle.layer.fc(
        input=[rnn_cell], size=vocab_size, act=paddle.activation.Softmax())

    # loss
    cost = paddle.layer.classification_cost(input=output, label=target)

    return cost, output


def train():
    """
    train rnn language model.

    :return: none, but this function will save the training model each epoch.
    """

    # load word dictionary
    print('load dictionary...')
    word_id_dict = reader.build_vocab()

    # define data reader
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.train_data(), buf_size=65536),
        batch_size=32)

    test_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.test_data(), buf_size=65536),
        batch_size=8)

    # network config
    print('prepare model...')
    cost, _ = lm(len(word_id_dict), emb_dim, rnn_type, hidden_size, num_layer)

    # create parameters
    parameters = paddle.parameters.create(cost)

    # create optimizer
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        regularization=paddle.optimizer.L2Regularization(rate=1e-3),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))

    # create trainer
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=adam_optimizer)

    # define event_handler callback
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print("\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost,
                    event.metrics))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

        # save model each pass
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_reader)
            print("\nTest with Pass %d, %s" % (event.pass_id, result.metrics))
            with gzip.open(model_file_name_prefix + str(event.pass_id) + '.tar.gz',
                           'w') as f:
                parameters.to_tar(f)

    # start to train
    print('start training...')

    trainer.train(
        reader=train_reader, event_handler=event_handler, num_passes=num_passs)

    print("Training finished.")

if __name__ == '__main__':

    # -- config --
    paddle.init(use_gpu=False, trainer_count=1)
    rnn_type = 'gru' # or 'lstm'
    emb_dim = 200
    hidden_size = 200
    num_passs = 2
    num_layer = 2
    model_file_name_prefix = 'lm_' + rnn_type + '_params_pass_'

    # -- train --
    train()

    # -- predict --

    # prepare model
    word_id_dict = reader.build_vocab() # load word dictionary
    _, output = lm(len(word_id_dict), emb_dim, rnn_type, hidden_size, num_layer) # network config
    model_file_name = model_file_name_prefix + str(num_passs - 1) + '.tar.gz'
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_file_name)) # load parameters
    # generate
    text = 'the end of'
    generate_sentences = generator.generate_with_beamSearch(output, parameters, word_id_dict, text, 5, 5)
    # print result
    for (sentence, prob) in generate_sentences.items():
        print(sentence.encode('utf-8', 'replace'))
        print('prob: ', prob)
        print('-------')
