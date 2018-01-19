# -*- coding: utf-8 -*-
import os
import io
import numpy as np

# Constants
UNK = "<UNK>"
SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<PAD>"
VOCAB_DIM = 2196017
EMBEDDING_DIM = 300
WORD2VEC = None


class Vocab(object):
    """Class to hold the vocabulary for the SquadDataset."""

    def __init__(self, path):
        self._id_to_word = []
        self._word_to_id = {}
        self._word_ending_tables = {}
        self._path = path
        self._pad = -1
        self._unk = None
        self._sos = None
        self._eos = None

        # first read in the base vocab
        with io.open(os.path.join(path, "vocab.txt"), "r") as f:
            for idx, line in enumerate(f):
                word_name = line.strip()
                if word_name == UNK:
                    self._unk = idx
                elif word_name == SOS:
                    self._sos = idx
                elif word_name == EOS:
                    self._eos = idx

                self._id_to_word.append(word_name)
                self._word_to_id[word_name] = idx

    @property
    def unk(self):
        return self._unk

    @property
    def sos(self):
        return self._sos

    @property
    def eos(self):
        return self._eos

    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_idx(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk

    def idx_to_word(self, idx):
        if idx == self._pad:
            return PAD
        if idx < self.size:
            return self._id_to_word[idx]
        return "ERROR"

    def decode(self, idxs):
        return " ".join([self.idx_to_word(idx) for idx in idxs])

    def encode(self, sentence):
        return [self.word_to_idx(word) for word in sentence]

    @property
    def word_embeddings(self):
        embedding_path = os.path.join(self._path, "embeddings.npy")
        embeddings = np.load(embedding_path)
        return embeddings

    def construct_embedding_matrix(self, glove_path):
        # Randomly initialize word embeddings
        embeddings = np.random.randn(self.size,
                                     EMBEDDING_DIM).astype(np.float32)

        load_word_vectors(
            param=embeddings,
            vocab=self._id_to_word,
            path=glove_path,
            missing_word_alternative=missing_word_heuristic,
            missing_word_value=lambda: 0.0)
        embedding_path = os.path.join(self._path, "embeddings.npy")
        np.save(embedding_path, embeddings)


def missing_word_heuristic(word, word2vec):
    """
    propose alternate spellings of a word to match against
    pretrained word vectors (so that if the original spelling
    has no pretrained vector, but alternate spelling does,
    a vector can be retrieved anyways.)
    """
    if len(word) > 5:
        # try to find similar words that share
        # the same 5 character ending:
        most_sim = word2vec.words_ending_in(word[-5:])

        if len(most_sim) > 0:
            most_sim = sorted(
                most_sim,
                reverse=True,
                key=lambda x: (
                    (word[0].isupper() == x[0].isupper()) +
                    (word.lower()[:3] == x.lower()[:3]) +
                    (word.lower()[:4] == x.lower()[:4]) +
                    (abs(len(word) - len(x)) < 5)
                )
            )
            return most_sim[:1]
    if all(not c.isalpha() for c in word):
        # this is a fully numerical answer (and non alpha)
        return ['13', '9', '100', '2.0']

    return [
        # add a capital letter
        word.capitalize(),
        # see if word has spurious period
        word.split(".")[0],
        # see if word has spurious backslash
        word.split("/")[0],
        # see if word has spurious parenthesis
        word.split(")")[0],
        word.split("(")[0]
    ]


class Word2Vec(object):
    """
    Load word2vec result from file
    """

    def __init__(self, vocab_size, vector_size):
        self.syn0 = np.zeros((vocab_size, vector_size), dtype=np.float32)
        self.index2word = []
        self.vocab_size = vocab_size
        self.vector_size = vector_size

    def load_word2vec_format(self, path):
        with io.open(path, "r") as fin:
            for word_id in range(self.vocab_size):
                line = fin.readline()
                parts = line.rstrip("\n").rstrip().split(" ")
                if len(parts) != self.vector_size + 1:
                    raise ValueError("invalid vector on line {}".format(
                        word_id))
                word, weights = parts[0], [np.float32(x) for x in parts[1:]]
                self.syn0[word_id] = weights
                self.index2word.append(word)
        return self


class FastWord2vec(object):
    """
    Load word2vec model, cache the embedding matrix using numpy
    and memory-map it so that future loads are fast.
    """

    def __init__(self, path):
        if not os.path.exists(path + ".npy"):
            word2vec = Word2Vec(VOCAB_DIM,
                                EMBEDDING_DIM).load_word2vec_format(path)

            # save as numpy
            np.save(path + ".npy", word2vec.syn0)
            # also save the vocab
            with io.open(path + ".vocab", "w", encoding="utf8") as fout:
                for word in word2vec.index2word:
                    fout.write(word + "\n")

        self.syn0 = np.load(path + ".npy", mmap_mode="r")
        self.index2word = [l.strip("\n") for l in io.open(path + ".vocab", "r")]
        self.word2index = {word: k for k, word in enumerate(self.index2word)}
        self._word_ending_tables = {}
        self._word_beginning_tables = {}

    def __getitem__(self, key):
        return np.array(self.syn0[self.word2index[key]])

    def __contains__(self, key):
        return key in self.word2index

    def words_ending_in(self, word_ending):
        if len(word_ending) == 0:
            return self.index2word
        self._build_word_ending_table(len(word_ending))
        return self._word_ending_tables[len(word_ending)].get(word_ending, [])

    def _build_word_ending_table(self, length):
        if length not in self._word_ending_tables:
            table = {}
            for word in self.index2word:
                if len(word) >= length:
                    ending = word[-length:]
                    if ending not in table:
                        table[ending] = [word]
                    else:
                        table[ending].append(word)
            self._word_ending_tables[length] = table

    def words_starting_in(self, word_beginning):
        if len(word_beginning) == 0:
            return self.index2word
        self._build_word_beginning_table(len(word_beginning))
        return self._word_beginning_tables[len(word_beginning)].get(
            word_beginning, [])

    def _build_word_beginning_table(self, length):
        if length not in self._word_beginning_tables:
            table = {}
            for word in get_progress_bar('building prefix lookup ')(
                    self.index2word):
                if len(word) >= length:
                    ending = word[:length]
                    if ending not in table:
                        table[ending] = [word]
                    else:
                        table[ending].append(word)
            self._word_beginning_tables[length] = table

    @staticmethod
    def get(path):
        global WORD2VEC
        if WORD2VEC is None:
            WORD2VEC = FastWord2vec(path)
        return WORD2VEC


def load_word_vectors(param,
                      vocab,
                      path,
                      verbose=True,
                      missing_word_alternative=None,
                      missing_word_value=None):
    """
    Add the pre-trained word embeddings stored under path to the parameter
    matrix `param` that has size `vocab x embedding_dim`.
    Arguments:
        param : np.array
        vocab : list<str>
        path : str, location of the pretrained word embeddings
        verbose : (optional) bool, whether to print how
            many words were recovered
    """
    word2vec = FastWord2vec.get(path)
    missing = 0
    for idx, word in enumerate(vocab):
        try:
            param[idx, :] = word2vec[word]
        except KeyError:
            try:
                param[idx, :] = word2vec[word.lower()]
            except KeyError:
                found = False
                if missing_word_alternative is not None:
                    alternatives = missing_word_alternative(word, word2vec)
                    if isinstance(alternatives, str):
                        alternatives = [alternatives]
                    assert (isinstance(alternatives, list)), (
                        "missing_word_alternative should return a list of strings."
                    )
                    for alternative in alternatives:
                        if alternative in word2vec:
                            param[idx, :] = word2vec[alternative]
                            found = True
                            break
                if not found:
                    if missing_word_value is not None:
                        param[idx, :] = missing_word_value()
                    missing += 1
    if verbose:
        print("Loaded {} words, {} missing".format(
            len(vocab) - missing, missing))
