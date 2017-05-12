import math
import gzip
import paddle.v2 as paddle
import paddle.v2.evaluator as evaluator
import conll03
import itertools

# init dataset
train_data_file = 'data/train'
test_data_file = 'data/test'
vocab_file = 'data/vocab.txt'
target_file = 'data/target.txt'
emb_file = 'data/wordVectors.txt'

word_dict, label_dict = conll03.get_dict(vocab_file, target_file)
word_vector_values = conll03.get_embedding(emb_file)
train_data_reader = conll03.train(train_data_file, vocab_file, target_file)
test_data_reader = conll03.test(test_data_file, vocab_file, target_file)

# init hyper-params
word_dict_len = len(word_dict)
label_dict_len = len(label_dict)
word_dim = 50
hidden_dim = 300

mix_hidden_lr = 1e-3
default_std = 1 / math.sqrt(hidden_dim) / 3.0
emb_para = paddle.attr.Param(
    name='emb', initial_std=math.sqrt(1. / word_dim), is_static=True)
std_0 = paddle.attr.Param(initial_std=0.)
std_default = paddle.attr.Param(initial_std=default_std)


def d_type(size):
    return paddle.data_type.integer_value_sequence(size)


def ner_net(is_train):
    word = paddle.layer.data(name='word', type=d_type(word_dict_len))

    word_embedding = paddle.layer.mixed(
        name='word_embedding',
        size=word_dim,
        input=paddle.layer.table_projection(input=word, param_attr=emb_para))
    emb_layers = [word_embedding]

    word_caps_vector = paddle.layer.concat(
        name='word_caps_vector', input=emb_layers)
    hidden_1 = paddle.layer.mixed(
        name='hidden1',
        size=hidden_dim,
        act=paddle.activation.Tanh(),
        bias_attr=std_default,
        input=[
            paddle.layer.full_matrix_projection(
                input=word_caps_vector, param_attr=std_default)
        ])

    rnn_para_attr = paddle.attr.Param(initial_std=0.0, learning_rate=0.1)
    hidden_para_attr = paddle.attr.Param(
        initial_std=default_std, learning_rate=mix_hidden_lr)

    rnn_1_1 = paddle.layer.recurrent(
        name='rnn1-1',
        input=hidden_1,
        act=paddle.activation.Relu(),
        bias_attr=std_0,
        param_attr=rnn_para_attr)
    rnn_1_2 = paddle.layer.recurrent(
        name='rnn1-2',
        input=hidden_1,
        act=paddle.activation.Relu(),
        reverse=1,
        bias_attr=std_0,
        param_attr=rnn_para_attr)

    hidden_2_1 = paddle.layer.mixed(
        name='hidden2-1',
        size=hidden_dim,
        bias_attr=std_default,
        act=paddle.activation.STanh(),
        input=[
            paddle.layer.full_matrix_projection(
                input=hidden_1, param_attr=hidden_para_attr),
            paddle.layer.full_matrix_projection(
                input=rnn_1_1, param_attr=rnn_para_attr)
        ])
    hidden_2_2 = paddle.layer.mixed(
        name='hidden2-2',
        size=hidden_dim,
        bias_attr=std_default,
        act=paddle.activation.STanh(),
        input=[
            paddle.layer.full_matrix_projection(
                input=hidden_1, param_attr=hidden_para_attr),
            paddle.layer.full_matrix_projection(
                input=rnn_1_2, param_attr=rnn_para_attr)
        ])

    rnn_2_1 = paddle.layer.recurrent(
        name='rnn2-1',
        input=hidden_2_1,
        act=paddle.activation.Relu(),
        reverse=1,
        bias_attr=std_0,
        param_attr=rnn_para_attr)
    rnn_2_2 = paddle.layer.recurrent(
        name='rnn2-2',
        input=hidden_2_2,
        act=paddle.activation.Relu(),
        bias_attr=std_0,
        param_attr=rnn_para_attr)

    hidden_3 = paddle.layer.mixed(
        name='hidden3',
        size=hidden_dim,
        bias_attr=std_default,
        act=paddle.activation.STanh(),
        input=[
            paddle.layer.full_matrix_projection(
                input=hidden_2_1, param_attr=hidden_para_attr),
            paddle.layer.full_matrix_projection(
                input=rnn_2_1,
                param_attr=rnn_para_attr), paddle.layer.full_matrix_projection(
                    input=hidden_2_2, param_attr=hidden_para_attr),
            paddle.layer.full_matrix_projection(
                input=rnn_2_2, param_attr=rnn_para_attr)
        ])

    output = paddle.layer.mixed(
        name='output',
        size=label_dict_len,
        bias_attr=False,
        input=[
            paddle.layer.full_matrix_projection(
                input=hidden_3, param_attr=std_default)
        ])

    if is_train:
        target = paddle.layer.data(name='target', type=d_type(label_dict_len))

        crf_cost = paddle.layer.crf(
            size=label_dict_len,
            input=output,
            label=target,
            param_attr=paddle.attr.Param(
                name='crfw',
                initial_std=default_std,
                learning_rate=mix_hidden_lr))

        crf_dec = paddle.layer.crf_decoding(
            size=label_dict_len,
            input=output,
            label=target,
            param_attr=paddle.attr.Param(name='crfw'))

        return crf_cost, crf_dec, target
    else:
        predict = paddle.layer.crf_decoding(
            size=label_dict_len,
            input=output,
            param_attr=paddle.attr.Param(name='crfw'))

        return predict


def ner_net_train(data_reader=train_data_reader, num_passes=1):
    # define network topology
    crf_cost, crf_dec, target = ner_net(is_train=True)
    evaluator.sum(name='error', input=crf_dec)

    # create parameters
    parameters = paddle.parameters.create(crf_cost)
    parameters.set('emb', word_vector_values)

    # create optimizer
    optimizer = paddle.optimizer.Momentum(
        momentum=0,
        learning_rate=2e-4,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        gradient_clipping_threshold=25,
        model_average=paddle.optimizer.ModelAverage(
            average_window=0.5, max_average_window=10000), )

    trainer = paddle.trainer.SGD(
        cost=crf_cost,
        parameters=parameters,
        update_equation=optimizer,
        extra_layers=crf_dec)

    reader = paddle.batch(
        paddle.reader.shuffle(data_reader, buf_size=8192), batch_size=64)

    feeding = {'word': 0, 'target': 1}

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            if event.batch_id % 1000 == 0:
                result = trainer.test(reader=reader, feeding=feeding)
                print "\nTest with Pass %d, Batch %d, %s" % (
                    event.pass_id, event.batch_id, result.metrics)

        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with gzip.open('params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
                parameters.to_tar(f)
            if event.pass_id == num_passes - 1:
                with gzip.open('ner_model.tar.gz', 'w') as f:
                    parameters.to_tar(f)
            result = trainer.test(reader=reader, feeding=feeding)
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

    trainer.train(
        reader=reader,
        event_handler=event_handler,
        num_passes=num_passes,
        feeding=feeding)

    return parameters


def ner_net_infer(data_reader=test_data_reader, model_file='ner_model.tar.gz'):
    test_creator = data_reader
    test_data = []
    test_sentences = []
    for item in test_creator():
        test_data.append([item[0]])
        test_sentences.append(item[-1])
        if len(test_data) == 10:
            break

    predict = ner_net(is_train=False)

    lab_ids = paddle.infer(
        output_layer=predict,
        parameters=paddle.parameters.Parameters.from_tar(gzip.open(model_file)),
        input=test_data,
        field='id')
    '''words_reverse = {}
    for (k, v) in word_dict.items():
        words_reverse[v] = k
    flat_data = [words_reverse[word_id] for word_id in itertools.chain.from_iterable(itertools.chain.from_iterable(test_data))]'''
    flat_data = [word for word in itertools.chain.from_iterable(test_sentences)]

    labels_reverse = {}
    for (k, v) in label_dict.items():
        labels_reverse[v] = k
    pre_lab = [labels_reverse[lab_id] for lab_id in lab_ids]

    for word, label in zip(flat_data, pre_lab):
        print word, label


if __name__ == '__main__':
    paddle.init(use_gpu=False, trainer_count=1)
    ner_net_train(num_passes=1)
    ner_net_infer()
