import sys
import six
import collections
import os
import sys
import io
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf-8')


def word_count(input_file, word_freq=None):
    """
    compute word count from corpus
    """
    if word_freq is None:
        word_freq = collections.defaultdict(int)

    for l in input_file:
        for w in l.strip().split():
            word_freq[w] += 1

    return word_freq


def build_dict(min_word_freq=0, train_dir="", test_dir=""):
    """
    Build a word dictionary from the corpus,  Keys of the dictionary are words,
    and values are zero-based IDs of these words.
    """
    word_freq = collections.defaultdict(int)
    files = os.listdir(train_dir)
    for fi in files:
        with io.open(os.path.join(train_dir, fi), "r") as f:
            word_freq = word_count(f, word_freq)
    files = os.listdir(test_dir)
    for fi in files:
        with io.open(os.path.join(test_dir, fi), "r") as f:
            word_freq = word_count(f, word_freq)

    word_freq = [x for x in six.iteritems(word_freq) if x[1] > min_word_freq]
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*word_freq_sorted))
    word_idx = dict(list(zip(words, six.moves.range(len(words)))))
    return word_idx


def write_paddle(word_idx, train_dir, test_dir, output_train_dir,
                 output_test_dir):
    files = os.listdir(train_dir)
    if not os.path.exists(output_train_dir):
        os.mkdir(output_train_dir)
    for fi in files:
        with io.open(os.path.join(train_dir, fi), "r") as f:
            with io.open(os.path.join(output_train_dir, fi), "w") as wf:
                for l in f:
                    l = l.strip().split()
                    l = [word_idx.get(w) for w in l]
                    for w in l:
                        wf.write(str2file(str(w) + " "))
                    wf.write(str2file("\n"))

    files = os.listdir(test_dir)
    if not os.path.exists(output_test_dir):
        os.mkdir(output_test_dir)
    for fi in files:
        with io.open(os.path.join(test_dir, fi), "r", encoding='utf-8') as f:
            with io.open(
                    os.path.join(output_test_dir, fi), "w",
                    encoding='utf-8') as wf:
                for l in f:
                    l = l.strip().split()
                    l = [word_idx.get(w) for w in l]
                    for w in l:
                        wf.write(str2file(str(w) + " "))
                    wf.write(str2file("\n"))


def str2file(str):
    if six.PY2:
        return str.decode("utf-8")
    else:
        return str


def text2paddle(train_dir, test_dir, output_train_dir, output_test_dir,
                output_vocab):
    vocab = build_dict(0, train_dir, test_dir)
    print("vocab size:", str(len(vocab)))
    with io.open(output_vocab, "w", encoding='utf-8') as wf:
        wf.write(str2file(str(len(vocab)) + "\n"))
    write_paddle(vocab, train_dir, test_dir, output_train_dir, output_test_dir)


train_dir = sys.argv[1]
test_dir = sys.argv[2]
output_train_dir = sys.argv[3]
output_test_dir = sys.argv[4]
output_vocab = sys.argv[5]
text2paddle(train_dir, test_dir, output_train_dir, output_test_dir,
            output_vocab)
