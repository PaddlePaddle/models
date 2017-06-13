# coding=utf-8
import sys
import paddle.v2 as paddle
import reader
from utils import *
import network_conf
import gzip
from config import *


def train(model_cost, train_reader, test_reader, model_file_name_prefix,
          num_passes):
    """
    train model.

    :param model_cost: cost layer of the model to train.
    :param train_reader: train data reader.
    :param test_reader: test data reader.
    :param model_file_name_prefix: model's prefix name.
    :param num_passes: epoch.
    :return:
    """

    # init paddle
    paddle.init(use_gpu=use_gpu, trainer_count=trainer_count)

    # create parameters
    parameters = paddle.parameters.create(model_cost)

    # create optimizer
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        regularization=paddle.optimizer.L2Regularization(rate=1e-3),
        model_average=paddle.optimizer.ModelAverage(
            average_window=0.5, max_average_window=10000))

    # create trainer
    trainer = paddle.trainer.SGD(
        cost=model_cost, parameters=parameters, update_equation=adam_optimizer)

    # define event_handler callback
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print("\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

        # save model each pass
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_reader)
            print("\nTest with Pass %d, %s" % (event.pass_id, result.metrics))
            with gzip.open(
                    model_file_name_prefix + str(event.pass_id) + '.tar.gz',
                    'w') as f:
                parameters.to_tar(f)

    # start to train
    print('start training...')
    trainer.train(
        reader=train_reader, event_handler=event_handler, num_passes=num_passes)

    print("Training finished.")


def main():
    # prepare vocab
    print('prepare vocab...')
    if build_vocab_method == 'fixed_size':
        word_id_dict = build_vocab_with_fixed_size(
            train_file, vocab_max_size)  # build vocab
    else:
        word_id_dict = build_vocab_using_threshhold(
            train_file, unk_threshold)  # build vocab
    save_vocab(word_id_dict, vocab_file)  # save vocab

    # init model and data reader
    if use_which_model == 'rnn':
        # init RNN model
        print('prepare rnn model...')
        config = Config_rnn()
        cost, _ = network_conf.rnn_lm(
            len(word_id_dict), config.emb_dim, config.rnn_type,
            config.hidden_size, config.num_layer)

        # init RNN data reader
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.rnn_reader(train_file, min_sentence_length,
                                  max_sentence_length, word_id_dict),
                buf_size=65536),
            batch_size=config.batch_size)

        test_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.rnn_reader(test_file, min_sentence_length,
                                  max_sentence_length, word_id_dict),
                buf_size=65536),
            batch_size=config.batch_size)

    elif use_which_model == 'ngram':
        # init N-Gram model
        print('prepare ngram model...')
        config = Config_ngram()
        assert config.N == 5
        cost, _ = network_conf.ngram_lm(
            vocab_size=len(word_id_dict),
            emb_dim=config.emb_dim,
            hidden_size=config.hidden_size,
            num_layer=config.num_layer)

        # init N-Gram data reader
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.ngram_reader(train_file, config.N, word_id_dict),
                buf_size=65536),
            batch_size=config.batch_size)

        test_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.ngram_reader(test_file, config.N, word_id_dict),
                buf_size=65536),
            batch_size=config.batch_size)
    else:
        raise Exception('use_which_model must be rnn or ngram!')

    # train model
    train(
        model_cost=cost,
        train_reader=train_reader,
        test_reader=test_reader,
        model_file_name_prefix=config.model_file_name_prefix,
        num_passes=config.num_passs)


if __name__ == "__main__":
    main()
