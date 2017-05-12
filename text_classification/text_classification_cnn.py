import sys
import paddle.v2 as paddle
import gzip


def convolution_net(dict_dim, class_dim=2, emb_dim=28, hid_dim=128):
    """
    cnn network definition

    :param dict_dim: size of word dictionary
    :type input_dim: int
    :params class_dim: number of instance class
    :type class_dim: int
    :params emb_dim: embedding vector dimension
    :type emb_dim: int
    :params hid_dim: number of same size convolution kernels
    :type hid_dim: int
    """

    # input layers
    data = paddle.layer.data("word",
                             paddle.data_type.integer_value_sequence(dict_dim))
    lbl = paddle.layer.data("label", paddle.data_type.integer_value(2))

    #embedding layer
    emb = paddle.layer.embedding(input=data, size=emb_dim)

    # convolution layers with max pooling
    conv_3 = paddle.networks.sequence_conv_pool(
        input=emb, context_len=3, hidden_size=hid_dim)
    conv_4 = paddle.networks.sequence_conv_pool(
        input=emb, context_len=4, hidden_size=hid_dim)

    # fc and output layer
    output = paddle.layer.fc(
        input=[conv_3, conv_4], size=class_dim, act=paddle.activation.Softmax())

    cost = paddle.layer.classification_cost(input=output, label=lbl)

    return cost, output, lbl


def train_cnn_model(num_pass):
    """
    train cnn model

    :params num_pass: train pass number
    :type num_pass: int
    """

    # load word dictionary
    print 'load dictionary...'
    word_dict = paddle.dataset.imdb.word_dict()

    dict_dim = len(word_dict)
    class_dim = 2
    # define data reader
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            lambda: paddle.dataset.imdb.train(word_dict), buf_size=1000),
        batch_size=100)
    test_reader = paddle.batch(
        lambda: paddle.dataset.imdb.test(word_dict), batch_size=100)

    # network config
    [cost, output, label] = convolution_net(dict_dim, class_dim=class_dim)
    # create parameters
    parameters = paddle.parameters.create(cost)
    # create optimizer
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        regularization=paddle.optimizer.L2Regularization(rate=1e-3),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))

    # add auc evaluator
    paddle.evaluator.auc(input=output, label=label)

    # create trainer
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=adam_optimizer)

    # Define end batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_reader, feeding=feeding)
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)
            with gzip.open("cnn_params_pass" + str(event.pass_id) + ".tar.gz",
                           'w') as f:
                parameters.to_tar(f)

    # begin training network
    feeding = {'word': 0, 'label': 1}
    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        feeding=feeding,
        num_passes=num_pass)

    print("Training finished.")


def cnn_infer(file_name):
    """
    predict instance labels by cnn network

    :params file_name: network parameter file
    :type file_name: str
    """

    print("Begin to predict...")

    word_dict = paddle.dataset.imdb.word_dict()
    dict_dim = len(word_dict)
    class_dim = 2

    [_, output, _] = convolution_net(dict_dim, class_dim=class_dim)
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(file_name))

    infer_data = []
    infer_data_label = []
    for item in paddle.dataset.imdb.test(word_dict):
        infer_data.append([item[0]])
        infer_data_label.append(item[1])

    predictions = paddle.infer(
        output_layer=output,
        parameters=parameters,
        input=infer_data,
        field=['value'])
    for i, prob in enumerate(predictions):
        print prob, infer_data_label[i]


if __name__ == "__main__":
    paddle.init(use_gpu=False, trainer_count=1)
    num_pass = 5
    train_cnn_model(num_pass=num_pass)
    param_file_name = "cnn_params_pass" + str(num_pass - 1) + ".tar.gz"
    cnn_infer(file_name=param_file_name)
