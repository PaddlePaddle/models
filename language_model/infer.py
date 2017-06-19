# coding=utf-8
import paddle.v2 as paddle
import gzip
import numpy as np
from utils import *
import network_conf
from config import *


def generate_using_rnn(word_id_dict, num_words, beam_size):
    """
    Demo: use RNN model to do prediction.

    :param word_id_dict: vocab.
    :type word_id_dict: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    :param num_words: the number of the words to generate.
    :type num_words: int
    :param beam_size: beam width.
    :type beam_size: int
    :return: save prediction results to output_file
    """

    # prepare and cache model
    config = Config_rnn()
    _, output_layer = network_conf.rnn_lm(
        vocab_size=len(word_id_dict),
        emb_dim=config.emb_dim,
        rnn_type=config.rnn_type,
        hidden_size=config.hidden_size,
        num_layer=config.num_layer)  # network config
    model_file_name = config.model_file_name_prefix + str(config.num_passs -
                                                          1) + '.tar.gz'
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(model_file_name))  # load parameters
    inferer = paddle.inference.Inference(
        output_layer=output_layer, parameters=parameters)

    # tools, different from generate_using_ngram's tools
    id_word_dict = dict(
        [(v, k) for k, v in word_id_dict.items()])  # {id : word}

    def str2ids(str):
        return [[[
            word_id_dict.get(w, word_id_dict['<UNK>']) for w in str.split()
        ]]]

    def ids2str(ids):
        return [[[id_word_dict.get(id, ' ') for id in ids]]]

    # generate text
    with open(input_file) as file:
        output_f = open(output_file, 'w')
        for line in file:
            line = line.decode('utf-8').strip()
            # generate
            texts = {}  # type: {text : probability}
            texts[line] = 1
            for _ in range(num_words):
                texts_new = {}
                for (text, prob) in texts.items():
                    if '<EOS>' in text:  # stop prediction when <EOS> appear
                        texts_new[text] = prob
                        continue
                    # next word's probability distribution
                    predictions = inferer.infer(input=str2ids(text))
                    predictions[-1][word_id_dict['<UNK>']] = -1  # filter <UNK>
                    # find next beam_size words
                    for _ in range(beam_size):
                        cur_maxProb_index = np.argmax(
                            predictions[-1])  # next word's id
                        text_new = text + ' ' + id_word_dict[
                            cur_maxProb_index]  # text append next word
                        texts_new[text_new] = texts[text] * predictions[-1][
                            cur_maxProb_index]
                        predictions[-1][cur_maxProb_index] = -1
                texts.clear()
                if len(texts_new) <= beam_size:
                    texts = texts_new
                else:  # cutting
                    texts = dict(
                        sorted(
                            texts_new.items(), key=lambda d: d[1], reverse=True)
                        [:beam_size])

            # save results to output file
            output_f.write(line.encode('utf-8') + '\n')
            for (sentence, prob) in texts.items():
                output_f.write('\t' + sentence.encode('utf-8', 'replace') + '\t'
                               + str(prob) + '\n')
            output_f.write('\n')

        output_f.close()
    print('already saved results to ' + output_file)


def generate_using_ngram(word_id_dict, num_words, beam_size):
    """
    Demo: use N-Gram model to do prediction.

    :param word_id_dict: vocab.
    :type word_id_dict: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    :param num_words: the number of the words to generate.
    :type num_words: int
    :param beam_size: beam width.
    :type beam_size: int
    :return: save prediction results to output_file
    """

    # prepare and cache model
    config = Config_ngram()
    _, output_layer = network_conf.ngram_lm(
        vocab_size=len(word_id_dict),
        emb_dim=config.emb_dim,
        hidden_size=config.hidden_size,
        num_layer=config.num_layer)  # network config
    model_file_name = config.model_file_name_prefix + str(config.num_passs -
                                                          1) + '.tar.gz'
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(model_file_name))  # load parameters
    inferer = paddle.inference.Inference(
        output_layer=output_layer, parameters=parameters)

    # tools, different from generate_using_rnn's tools
    id_word_dict = dict(
        [(v, k) for k, v in word_id_dict.items()])  # {id : word}

    def str2ids(str):
        return [[
            word_id_dict.get(w, word_id_dict['<UNK>']) for w in str.split()
        ]]

    def ids2str(ids):
        return [[id_word_dict.get(id, ' ') for id in ids]]

    # generate text
    with open(input_file) as file:
        output_f = open(output_file, 'w')
        for line in file:
            line = line.decode('utf-8').strip()
            words = line.split()
            if len(words) < config.N:
                output_f.write(line.encode('utf-8') + "\n\tnone\n")
                continue
            # generate
            texts = {}  # type: {text : probability}
            texts[line] = 1
            for _ in range(num_words):
                texts_new = {}
                for (text, prob) in texts.items():
                    if '<EOS>' in text:  # stop prediction when <EOS> appear
                        texts_new[text] = prob
                        continue
                    # next word's probability distribution
                    predictions = inferer.infer(
                        input=str2ids(' '.join(text.split()[-config.N:])))
                    predictions[-1][word_id_dict['<UNK>']] = -1  # filter <UNK>
                    # find next beam_size words
                    for _ in range(beam_size):
                        cur_maxProb_index = np.argmax(
                            predictions[-1])  # next word's id
                        text_new = text + ' ' + id_word_dict[
                            cur_maxProb_index]  # text append nextWord
                        texts_new[text_new] = texts[text] * predictions[-1][
                            cur_maxProb_index]
                        predictions[-1][cur_maxProb_index] = -1
                texts.clear()
                if len(texts_new) <= beam_size:
                    texts = texts_new
                else:  # cutting
                    texts = dict(
                        sorted(
                            texts_new.items(), key=lambda d: d[1], reverse=True)
                        [:beam_size])

            # save results to output file
            output_f.write(line.encode('utf-8') + '\n')
            for (sentence, prob) in texts.items():
                output_f.write('\t' + sentence.encode('utf-8', 'replace') + '\t'
                               + str(prob) + '\n')
            output_f.write('\n')

        output_f.close()
    print('already saved results to ' + output_file)


def main():
    # init paddle
    paddle.init(use_gpu=use_gpu, trainer_count=trainer_count)

    # prepare and cache vocab
    if os.path.isfile(vocab_file):
        word_id_dict = load_vocab(vocab_file)  # load word dictionary
    else:
        if build_vocab_method == 'fixed_size':
            word_id_dict = build_vocab_with_fixed_size(
                train_file, vocab_max_size)  # build vocab
        else:
            word_id_dict = build_vocab_using_threshhold(
                train_file, unk_threshold)  # build vocab
        save_vocab(word_id_dict, vocab_file)  # save vocab

    # generate
    if use_which_model == 'rnn':
        generate_using_rnn(
            word_id_dict=word_id_dict, num_words=num_words, beam_size=beam_size)
    elif use_which_model == 'ngram':
        generate_using_ngram(
            word_id_dict=word_id_dict, num_words=num_words, beam_size=beam_size)
    else:
        raise Exception('use_which_model must be rnn or ngram!')


if __name__ == "__main__":
    main()
