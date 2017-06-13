# coding=utf-8
import os
import collections


def save_vocab(word_id_dict, vocab_file_name):
    """
    save vocab.

    :param word_id_dict: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    :param vocab_file_name: vocab file name.
    """
    f = open(vocab_file_name, 'w')
    for (k, v) in word_id_dict.items():
        f.write(k.encode('utf-8') + '\t' + str(v) + '\n')
    print('save vocab to ' + vocab_file_name)
    f.close()


def load_vocab(vocab_file_name):
    """
    load vocab from file.
    :param vocab_file_name: vocab file name.
    :return: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    """
    assert os.path.isfile(vocab_file_name)
    dict = {}
    with open(vocab_file_name) as file:
        for line in file:
            if len(line) < 2:
                continue
            kv = line.decode('utf-8').strip().split('\t')
            dict[kv[0]] = int(kv[1])
    return dict


def build_vocab_using_threshhold(file_name, unk_threshold):
    """
    build vacab using_<UNK> threshhold.

    :param file_name:
    :param unk_threshold: <UNK> threshhold.
    :type unk_threshold: int.
    :return: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    """
    counter = {}
    with open(file_name) as file:
        for line in file:
            words = line.decode('utf-8', 'ignore').strip().split()
            for word in words:
                if word in counter:
                    counter[word] += 1
                else:
                    counter[word] = 1
    counter_new = {}
    for (word, frequency) in counter.items():
        if frequency >= unk_threshold:
            counter_new[word] = frequency
    counter.clear()
    counter_new = sorted(counter_new.items(), key=lambda d: -d[1])
    words = [word_frequency[0] for word_frequency in counter_new]
    word_id_dict = dict(zip(words, range(2, len(words) + 2)))
    word_id_dict['<UNK>'] = 0
    word_id_dict['<EOS>'] = 1
    return word_id_dict


def build_vocab_with_fixed_size(file_name, vocab_max_size):
    """
    build vacab with assigned max size.

    :param vocab_max_size: vocab's max size.
    :return: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    """
    words = []
    for line in open(file_name):
        words += line.decode('utf-8', 'ignore').strip().split()

    counter = collections.Counter(words)
    counter = sorted(counter.items(), key=lambda x: -x[1])
    if len(counter) > vocab_max_size:
        counter = counter[:vocab_max_size]
    words, counts = zip(*counter)
    word_id_dict = dict(zip(words, range(2, len(words) + 2)))
    word_id_dict['<UNK>'] = 0
    word_id_dict['<EOS>'] = 1
    return word_id_dict
