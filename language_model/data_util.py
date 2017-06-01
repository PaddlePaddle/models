# coding=utf-8
import numpy as np
import collections

# config
train_file = 'data/ptb.train.txt'
test_file = 'data/ptb.test.txt'
vocab_max_size = 3000
min_sentence_length = 3
max_sentence_length = 60

def build_vocab():
    """
    build vacab.
    
    :return: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    """
    words = []
    for line in open(train_file):
        words += line.decode('utf-8','ignore').strip().split()

    counter = collections.Counter(words)
    counter = sorted(counter.items(), key=lambda x: -x[1])
    if len(counter) > vocab_max_size:
        counter = counter[:vocab_max_size]
    words, counts = zip(*counter)
    word_id_dict = dict(zip(words, range(2, len(words) + 2)))
    word_id_dict['<UNK>'] = 0
    word_id_dict['<EOS>'] = 1
    return word_id_dict

def _read_by_fixed_length(file_name, sentence_len=10):
    """
    create reader, each sample with fixed length.

    :param file_name: file name.
    :param sentence_len: each sample's length.
    :return: data reader.
    """
    def reader():
        word_id_dict = build_vocab()
        words = []
        UNK = word_id_dict['<UNK>']
        for line in open(file_name):
            words += line.decode('utf-8','ignore').strip().split()
        ids = [word_id_dict.get(w, UNK) for w in words]
        words_len = len(words)
        sentence_num = (words_len-1) // sentence_len
        count = 0
        while count < sentence_num:
            start = count * sentence_len
            count += 1
            yield ids[start:start+sentence_len], ids[start+1:start+sentence_len+1]
    return reader

def _read_by_line(file_name):
    """
    create reader, each line is a sample.

    :param file_name: file name.
    :return: data reader.
    """
    def reader():
        word_id_dict = build_vocab()
        UNK = word_id_dict['<UNK>']
        for line in open(file_name):
            words = line.decode('utf-8','ignore').strip().split()
            if len(words) < min_sentence_length or len(words) > max_sentence_length:
                continue
            ids = [word_id_dict.get(w, UNK) for w in words]
            ids.append(word_id_dict['<EOS>'])
            target = ids[1:]
            target.append(word_id_dict['<EOS>'])
            yield ids[:], target[:]
    return reader

def _reader_creator_for_NGram(file_name, N):
    """
    create reader for ngram.

    :param file_name: file name.
    :param N: ngram's n.
    :return: data reader.
    """
    assert N >= 2
    def reader():
        word_id_dict = build_vocab()
        words = []
        UNK = word_id_dict['<UNK>']
        for line in open(file_name):
            words += line.decode('utf-8','ignore').strip().split()
        ids = [word_id_dict.get(w, UNK) for w in words]
        words_len = len(words)
        for i in range(words_len-N-1):
            yield tuple(ids[i:i+N])
    return reader

def train_data():
    return _read_by_line(train_file)

def test_data():
    return _read_by_line(test_file)

def train_data_for_NGram(N):
    return _reader_creator_for_NGram(train_file, N)

def test_data_for_NGram(N):
    return _reader_creator_for_NGram(test_file, N)
