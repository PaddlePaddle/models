# coding=utf-8
import sys
import paddle.v2 as paddle
import data_util as reader
import gzip
import generate_text as generator

def lm(vocab_size, emb_dim, hidden_size, num_layer):
    """
    ngram language model definition.

    :param vocab_size: size of vocab.
    :param emb_dim: embedding vector's dimension.
    :param hidden_size: size of unit.
    :param num_layer: layer number.
    :return: cost and output of model.
    """

    assert emb_dim > 0 and hidden_size > 0 and vocab_size > 0 and num_layer > 0

    def wordemb(inlayer):
        wordemb = paddle.layer.table_projection(
            input=inlayer,
            size=emb_dim,
            param_attr=paddle.attr.Param(
                name="_proj",
                initial_std=0.001,
                learning_rate=1,
                l2_rate=0, ))
        return wordemb

    # input layers
    firstword = paddle.layer.data(
        name="firstw", type=paddle.data_type.integer_value(vocab_size))
    secondword = paddle.layer.data(
        name="secondw", type=paddle.data_type.integer_value(vocab_size))
    thirdword = paddle.layer.data(
        name="thirdw", type=paddle.data_type.integer_value(vocab_size))
    fourthword = paddle.layer.data(
        name="fourthw", type=paddle.data_type.integer_value(vocab_size))

    # embedding layer
    Efirst = wordemb(firstword)
    Esecond = wordemb(secondword)
    Ethird = wordemb(thirdword)
    Efourth = wordemb(fourthword)

    contextemb = paddle.layer.concat(input=[Efirst, Esecond, Ethird, Efourth])

    # hidden layer
    hidden = paddle.layer.fc(
        input=contextemb, size=hidden_size, act=paddle.activation.Relu())
    for _ in range(num_layer - 1):
        hidden = paddle.layer.fc(
            input=hidden, size=hidden_size, act=paddle.activation.Relu())

    # fc and output layer
    predictword = paddle.layer.fc(
        input=[hidden], size=vocab_size, act=paddle.activation.Softmax())

    # loss
    nextword = paddle.layer.data(
        name="fifthw", type=paddle.data_type.integer_value(vocab_size))
    cost = paddle.layer.classification_cost(input=predictword, label=nextword)

    return cost, predictword


def train():
    """
    train ngram language model.

    :return: none, but this function will save the  training model each epoch.
    """

    # load word dictionary
    print('load dictionary...')
    word_id_dict = reader.build_vocab()

    # define data reader
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.train_data_for_NGram(N), buf_size=65536),
        batch_size=32)

    test_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.test_data_for_NGram(N), buf_size=65536),
        batch_size=8)

    # network config
    print('prepare model...')
    cost, _ = lm(len(word_id_dict), emb_dim, hidden_size, num_layer)

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
    emb_dim = 200
    hidden_size = 200
    num_passs = 2
    num_layer = 2
    N = 5
    model_file_name_prefix = 'lm_ngram_pass_'

    # -- train --
    train()

    # -- predict --

    # prepare model
    word_id_dict = reader.build_vocab()  # load word dictionary
    _, output = lm(len(word_id_dict), emb_dim, hidden_size, num_layer)  # network config
    model_file_name =  model_file_name_prefix + str(num_passs - 1) + '.tar.gz'
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_file_name))  # load parameters
    # generate
    text = 'the end of the'
    input = [[word_id_dict.get(w, word_id_dict['<UNK>']) for w in text.split()]]
    print(generator.next_word(output, parameters, word_id_dict, input))

