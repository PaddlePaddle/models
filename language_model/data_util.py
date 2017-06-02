# coding=utf-8
import collections
import os

# -- function --

def save_vocab(word_id_dict, vocab_file_name):
    """
    save vocab.
    :param word_id_dict: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    :param vocab_file_name: vocab file name.
    """
    f = open(vocab_file_name,'w')
    for(k, v) in word_id_dict.items():
        f.write(k.encode('utf-8') + '\t' + str(v) + '\n')
    print('save vocab to '+vocab_file_name)
    f.close()

def load_vocab(vocab_file_name):
    """
    load vocab from file
    :param vocab_file_name: vocab file name.
    :return: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    """
    if not os.path.isfile(vocab_file_name):
        raise Exception('vocab file does not exist!')
    dict = {}
    for line in open(vocab_file_name):
        if len(line) < 2:
            continue
        kv = line.decode('utf-8').strip().split('\t')
        dict[kv[0]] = int(kv[1])
    return dict

def build_vocab(file_name, vocab_max_size):
    """
    build vacab.

    :param vocab_max_size: vocab's max size.
    :return: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    """
    words = []
    for line in open(file_name):
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

def _read_by_fixed_length(file_name, word_id_dict, sentence_len=10):
    """
    create reader, each sample with fixed length.

    :param file_name: file name.
    :param word_id_dict: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    :param sentence_len: each sample's length.
    :return: data reader.
    """
    def reader():
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

def _read_by_line(file_name, min_sentence_length, max_sentence_length, word_id_dict):
    """
    create reader, each line is a sample.

    :param file_name: file name.
    :param min_sentence_length: sentence's min length.
    :param max_sentence_length: sentence's max length.
    :param word_id_dict: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    :return: data reader.
    """
    def reader():
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

def _reader_creator_for_NGram(file_name, N, word_id_dict):
    """
    create reader for ngram.

    :param file_name: file name.
    :param N: ngram's n.
    :param word_id_dict: dictionary with content of '{word, id}', 'word' is string type , 'id' is int type.
    :return: data reader.
    """
    assert N >= 2
    def reader():
        words = []
        UNK = word_id_dict['<UNK>']
        for line in open(file_name):
            words += line.decode('utf-8','ignore').strip().split()
        ids = [word_id_dict.get(w, UNK) for w in words]
        words_len = len(words)
        for i in range(words_len-N-1):
            yield tuple(ids[i:i+N])
    return reader

def train_data(train_file, min_sentence_length, max_sentence_length, word_id_dict):
    return _read_by_line(train_file, min_sentence_length, max_sentence_length, word_id_dict)

def test_data(test_file, min_sentence_length, max_sentence_length, word_id_dict):
    return _read_by_line(test_file, min_sentence_length, max_sentence_length, word_id_dict)

def train_data_for_NGram(train_file, N, word_id_dict):
    return _reader_creator_for_NGram(train_file, N, word_id_dict)

def test_data_for_NGram(test_file, N, word_id_dict):
    return _reader_creator_for_NGram(test_file, N, word_id_dict)
