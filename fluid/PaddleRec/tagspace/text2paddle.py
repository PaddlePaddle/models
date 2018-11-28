import sys
import six
import collections
import os
import csv 
import re

def word_count(column_num, input_file, word_freq=None):
    """
    compute word count from corpus
    """
    if word_freq is None:
        word_freq = collections.defaultdict(int)
    data_file = csv.reader(input_file)
    for row in data_file:
        for w in re.split(r'\W+',row[column_num].strip()):
            word_freq[w]+= 1
    return word_freq

def build_dict(column_num=2, min_word_freq=0, train_dir="", test_dir=""):
    """
    Build a word dictionary from the corpus,  Keys of the dictionary are words,
    and values are zero-based IDs of these words.
    """
    word_freq = collections.defaultdict(int)
    files = os.listdir(train_dir)
    for fi in files:
        with open(train_dir + '/' + fi, "r") as f:
            word_freq = word_count(column_num, f, word_freq)
    files = os.listdir(test_dir)
    for fi in files:
        with open(test_dir + '/' + fi, "r") as f:
            word_freq = word_count(column_num, f, word_freq)

    word_freq = [x for x in six.iteritems(word_freq) if x[1] > min_word_freq]
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*word_freq_sorted))
    word_idx = dict(list(zip(words, six.moves.range(len(words)))))
    return word_idx


def write_paddle(text_idx, tag_idx, train_dir, test_dir, output_train_dir, output_test_dir):
    files = os.listdir(train_dir)
    if not os.path.exists(output_train_dir):
        os.mkdir(output_train_dir)
    for fi in files:
        with open(train_dir + '/' + fi, "r") as f:
            with open(output_train_dir + '/' + fi, "w") as wf:
                data_file = csv.reader(f)
                for row in data_file:
                    tag_raw = re.split(r'\W+', row[0].strip())
                    pos_index = tag_idx.get(tag_raw[0])
                    wf.write(str(pos_index) + ",")
                    text_raw = re.split(r'\W+', row[2].strip())
                    l = [text_idx.get(w) for w in text_raw]
                    for w in l:
                        wf.write(str(w) + " ")
                    wf.write("\n")

    files = os.listdir(test_dir)
    if not os.path.exists(output_test_dir):
        os.mkdir(output_test_dir)
    for fi in files:
        with open(test_dir + '/' + fi, "r") as f:
            with open(output_test_dir + '/' + fi, "w") as wf:
                data_file = csv.reader(f)
                for row in data_file:
                    tag_raw = re.split(r'\W+', row[0].strip())
                    pos_index = tag_idx.get(tag_raw[0])
                    wf.write(str(pos_index) + ",")
                    text_raw = re.split(r'\W+', row[2].strip())
                    l = [text_idx.get(w) for w in text_raw]
                    for w in l:
                        wf.write(str(w) + " ")
                    wf.write("\n")

def text2paddle(train_dir, test_dir, output_train_dir, output_test_dir, output_vocab_text, output_vocab_tag):
    print("start constuct word dict")
    vocab_text = build_dict(2, 0, train_dir, test_dir)
    with open(output_vocab_text, "w") as wf:
        wf.write(str(len(vocab_text)) + "\n")

    vocab_tag = build_dict(0, 0, train_dir, test_dir)
    with open(output_vocab_tag, "w") as wf:
        wf.write(str(len(vocab_tag)) + "\n")

    print("construct word dict done\n")
    write_paddle(vocab_text, vocab_tag, train_dir, test_dir, output_train_dir, output_test_dir)


train_dir = sys.argv[1]
test_dir = sys.argv[2]
output_train_dir = sys.argv[3]
output_test_dir = sys.argv[4]
output_vocab_text = sys.argv[5]
output_vocab_tag = sys.argv[6]
text2paddle(train_dir, test_dir, output_train_dir, output_test_dir, output_vocab_text, output_vocab_tag)
