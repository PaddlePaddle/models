# coding=utf-8
import sys
import paddle.v2 as paddle
import data_util as reader
import gzip
import os
import numpy as np


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

    # prepare word dictionary
    print('prepare vocab...')
    word_id_dict = reader.build_vocab(train_file, vocab_max_size)  # build vocab
    reader.save_vocab(word_id_dict, vocab_file)  # save vocab

    # define data reader
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.train_data(
                train_file, min_sentence_length,
                max_sentence_length, word_id_dict), buf_size=65536),
        batch_size=32)

    test_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.test_data(
                test_file, min_sentence_length, max_sentence_length, word_id_dict), buf_size=65536),
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


def _generate_with_beamSearch(inferer, word_id_dict, input, num_words, beam_size):
    """
    Demo: generate 'num_words' words using "beam search" algorithm.

    :param inferer: paddle's inferer
    :type inferer: paddle.inference.Inference
    :param word_id_dict: vocab.
    :type word_id_dict: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    :param input: prefix text.
    :type input: string.
    :param num_words: the number of the words to generate.
    :type num_words: int
    :param beam_size: beam with.
    :type beam_size: int
    :return: text with generated words. dictionary with content of '{text, probability}'
    """

    assert beam_size > 0 and num_words > 0

    # load word dictionary
    id_word_dict = dict([(v, k) for k, v in word_id_dict.items()])  # {id : word}

    # tools
    def str2ids(str):
        return [[[word_id_dict.get(w, word_id_dict['<UNK>']) for w in str.split()]]]

    def ids2str(ids):
        return [[[id_word_dict.get(id, ' ') for id in ids]]]

    # generate
    texts = {}  # type: {text : prob}
    texts[input] = 1
    for _ in range(num_words):
        texts_new = {}
        for (text, prob) in texts.items():
            # next word's prob distubution
            predictions = inferer.infer(input=str2ids(text))
            predictions[-1][word_id_dict['<UNK>']] = -1  # filter <UNK>
            # find next beam_size words
            for _ in range(beam_size):
                cur_maxProb_index = np.argmax(predictions[-1])  # next word's id
                text_new = text + ' ' + id_word_dict[cur_maxProb_index]  # text append nextWord
                texts_new[text_new] = texts[text] * predictions[-1][cur_maxProb_index]
                predictions[-1][cur_maxProb_index] = -1
        texts.clear()
        if len(texts_new) <= beam_size:
            texts = texts_new
        else:  # cutting
            texts = dict(sorted(texts_new.items(), key=lambda d: d[1], reverse=True)[:beam_size])

    return texts


def predict():
    """
    demo: use model to do prediction.

    :return: print result to console.
    """

    # prepare and cache vocab
    if os.path.isfile(vocab_file):
        word_id_dict = reader.load_vocab(vocab_file)  # load word dictionary
    else:
        word_id_dict = reader.build_vocab(train_file, vocab_max_size)  # build vocab
        reader.save_vocab(word_id_dict, vocab_file)  # save vocab

    # prepare and cache model
    _, output = lm(len(word_id_dict), emb_dim, rnn_type, hidden_size, num_layer)  # network config
    model_file_name = model_file_name_prefix + str(num_passs - 1) + '.tar.gz'
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_file_name))  # load parameters
    inferer = paddle.inference.Inference(output_layer=output, parameters=parameters)

    # generate text
    while True:
        input_str = raw_input('input:')
        input_str_uft8 = input_str.decode('utf-8')
        generate_sentences = _generate_with_beamSearch(
            inferer=inferer, word_id_dict=word_id_dict, input=input_str_uft8, num_words=5, beam_size=5)
        # print result
        for (sentence, prob) in generate_sentences.items():
            print(sentence.encode('utf-8', 'replace'))
            print('prob: ', prob)
            print('-------')


if __name__ == '__main__':
    # -- config : model --
    rnn_type = 'gru'  # or 'lstm'
    emb_dim = 200
    hidden_size = 200
    num_passs = 2
    num_layer = 2
    model_file_name_prefix = 'lm_' + rnn_type + '_params_pass_'

    # -- config : data --
    train_file = 'data/ptb.train.txt'
    test_file = 'data/ptb.test.txt'
    vocab_file = 'data/vocab_ptb.txt'  # the file to save vocab
    vocab_max_size = 3000
    min_sentence_length = 3
    max_sentence_length = 60

    # -- train --
    paddle.init(use_gpu=False, trainer_count=1)
    train()

    # -- predict --
    predict()
