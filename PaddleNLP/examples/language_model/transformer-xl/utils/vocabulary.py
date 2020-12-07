import os
import numpy as np
from collections import Counter, OrderedDict


class Vocab(object):
    def __init__(self,
                 special=[],
                 min_freq=0,
                 max_size=None,
                 lower_case=True,
                 delimiter=None,
                 vocab_file=None):
        for arg, value in locals().items():
            if arg != "self":
                setattr(self, "_" + arg, value)
        self._counter = Counter()

    def __len__(self):
        return len(self.idx2sym)

    def tokenize(self, line, add_eos=False, add_double_eos=False):
        line = line.strip()
        if self._lower_case:
            line = line.lower()

        if self._delimiter == "":
            symbols = line
        else:
            symbols = line.split(self._delimiter)

        if add_double_eos:
            # Used for lm1b dataset.
            return ['<s>'] + symbols + ['<s>']
        elif add_eos:
            return symbols + ['<eos>']
        else:
            return symbols

    def cnt_file(self, path, add_eos=False):
        assert os.path.exists(path), "%s is not exist. " % path

        sentences = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                symbols = self.tokenize(line, add_eos=add_eos)
                self._counter.update(symbols)
                sentences.append(symbols)

        return sentences

    def cnt_sentences(self, sentences):
        for symbols in sentences:
            self._counter.update(symbols)

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        self._unk_idx = self.sym2idx['<unk>']

    def build_vocab(self):
        if self._vocab_file:
            self._build_from_file(self._vocab_file)
        else:
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            for symbol in self._special:
                self.add_special(symbol)

            for symbol, cnt in self._counter.most_common(self._max_size):
                if cnt < self._min_freq:
                    break
                self.add_symbol(symbol)

    def encode_file(self,
                    path,
                    ordered=False,
                    add_eos=True,
                    add_double_eos=False):
        assert os.path.exists(path), "%s is not exist. " % path

        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                symbols = self.tokenize(
                    line, add_eos=add_eos, add_double_eos=add_double_eos)
                encoded.append(
                    np.asarray(self.get_indices(symbols)).astype("int64"))

        if ordered:
            encoded = np.concatenate(encoded)

        return encoded

    def encode_sentences(self, sentences, ordered=False):
        encoded = []
        for symbols in sentences:
            encoded.append(
                np.asarray(self.get_indices(symbols)).astype("int64"))

        if ordered:
            encoded = np.concatenate(encoded)

        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, "_%s_idx" % (sym.strip('<>')), self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_sym(self, index):
        assert 0 <= index < len(self), 'Index %d out of range' % index
        return self.idx2sym[index]

    def get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            assert '<eos>' not in sym
            assert hasattr(self, '_unk_idx')
            return self.sym2idx.get(sym, self._unk_idx)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_sentences(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self.get_sym(idx) for idx in indices])
        else:
            return ' '.join(
                [self.get_sym(idx) for idx in indices if idx not in exclude])
