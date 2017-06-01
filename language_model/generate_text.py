# coding=utf-8
import paddle.v2 as paddle
import numpy as np

def next_word(model_struct, model_params, word_id_dict, input):
    """
    Demo: generate the next word.
        to show the simplest way using trained model to do prediction.

    :param model_struct: model's structure, only the output layer will be used for prediction task.
    :param model_params: parameters trained before.
    :param word_id_dict: vocab.
    :type word_id_dict: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    :param input: input.
    :type input: integer sequence.
    :return: predict word.
    """

    predictions = paddle.infer(
        output_layer=model_struct,
        parameters=model_params,
        input=input,
        field=['value'])

    id_word_dict = dict([(v, k) for k, v in word_id_dict.items()])  # dictionary with type {id : word}
    predictions[-1][word_id_dict['<UNK>']] = -1  # filter <UNK>
    return id_word_dict[np.argmax(predictions[-1])]


def generate_with_greedy(model_struct, model_params, word_id_dict, text, num_words):
    """
    Demo: generate 'num_words' words using greedy algorithm.

    :param model_struct: model's structure, only the output layer will be used for prediction task.
    :param model_params: parameters trained before.
    :param word_id_dict: vocab.
    :type word_id_dict: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    :param text: prefix text.
    :type text: string.
    :param num_words: the number of the words to generate.
    :return: text with generated words.
    """

    assert num_words > 0

    # prepare dictionary
    id_word_dict = dict([(v, k) for k, v in word_id_dict.items()])

    # generate
    for _ in range(num_words):
        text_ids = [[[word_id_dict.get(w, word_id_dict['<UNK>']) for w in text.split()]]]
        print('input:', text.encode('utf-8', 'replace'), text_ids)
        predictions = paddle.infer(
            output_layer=model_struct,
            parameters=model_params,
            input=text_ids,
            field=['value'])
        predictions[-1][word_id_dict['<UNK>']] = -1  # filter <UNK>
        text += ' ' + id_word_dict[np.argmax(predictions[-1])]

    return text


def generate_with_beamSearch(model_struct, model_params, word_id_dict, text, num_words, beam_size):
    """
    Demo: generate 'num_words' words using "beam search" algorithm.

    :param model_struct: model's structure, only the output layer will be used for prediction task.
    :param model_params: parameters trained before.
    :param word_id_dict: vocab.
    :type word_id_dict: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    :param text: prefix text.
    :type text: string.
    :param num_words: the number of the words to generate.
    :param beam_size: beam with.
    :return: text with generated words.
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
    texts[text] = 1
    for _ in range(num_words):
        texts_new = {}
        for (text, prob) in texts.items():
            # next word's prob distubution
            predictions = paddle.infer(
                output_layer=model_struct,
                parameters=model_params,
                input=str2ids(text),
                field=['value'])
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

