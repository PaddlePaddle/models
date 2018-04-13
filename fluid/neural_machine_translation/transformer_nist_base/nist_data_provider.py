import os
from functools import partial
from collections import defaultdict

__all__ = [
    "train",
    "test",
    "get_dict",
]

DATA_HOME = "/root/data/nist06n/"

START_MARK = "_GO"
END_MARK = "_EOS"
UNK_MARK = "_UNK"


def __build_dict(data_file, dict_size, save_path, lang="cn"):
    word_dict = defaultdict(int)
    data_files = [os.path.join(data_file, f) for f in os.listdir(data_file)
                  ] if os.path.isdir(data_file) else [data_file]

    for file_path in data_files:
        with open(file_path, mode="r") as f:
            for line in f.readlines():
                line_split = line.strip().split("\t")
                if len(line_split) != 2: continue
                sen = line_split[0] if lang == "cn" else line_split[1]
                for w in sen.split():
                    word_dict[w] += 1

    with open(save_path, "w") as fout:
        fout.write("%s\n%s\n%s\n" % (START_MARK, END_MARK,
                                         UNK_MARK))
        for idx, word in enumerate(
                sorted(word_dict.iteritems(), key=lambda x: x[1],
                       reverse=True)):
            if idx + 3 == dict_size: break
            fout.write("%s\n" % (word[0]))


def __load_dict(data_file, dict_size, lang, dict_file=None, reverse=False):
    dict_file = "%s_%d.dict" % (lang,
                                dict_size) if dict_file is None else dict_file
    dict_path = os.path.join(DATA_HOME, dict_file)
    data_path = os.path.join(DATA_HOME, data_file)
    if not os.path.exists(dict_path) or (len(open(dict_path, "r").readlines())
                                         != dict_size):
        __build_dict(data_path, dict_size, dict_path, lang)

    word_dict = {}
    with open(dict_path, "r") as fdict:
        for idx, line in enumerate(fdict):
            if reverse:
                word_dict[idx] = line.strip()
            else:
                word_dict[line.strip()] = idx
    return word_dict


def reader_creator(data_file,
                   src_lang,
                   src_dict_size,
                   trg_dict_size,
                   src_dict_file=None,
                   trg_dict_file=None,
                   len_filter=200):
    def reader():
        src_dict = __load_dict(data_file, src_dict_size, "cn", src_dict_file)
        trg_dict = __load_dict(data_file, trg_dict_size, "en", trg_dict_file)

        # the indice for start mark, end mark, and unk are the same in source
        # language and target language. Here uses the source language
        # dictionary to determine their indices.
        start_id = src_dict[START_MARK]
        end_id = src_dict[END_MARK]
        unk_id = src_dict[UNK_MARK]

        src_col = 0 if src_lang == "cn" else 1
        trg_col = 1 - src_col

        data_path = os.path.join(DATA_HOME, data_file)
        data_files = [
            os.path.join(data_path, f) for f in os.listdir(data_path)
        ] if os.path.isdir(data_path) else [data_path]
        for file_path in data_files:
            with open(file_path, mode="r") as f:
                for line in f.readlines():
                    line_split = line.strip().split("\t")
                    if len(line_split) != 2:
                        continue
                    src_words = line_split[src_col].split()
                    src_ids = [start_id
                               ] + [src_dict.get(w, unk_id)
                                    for w in src_words] + [end_id]

                    trg_words = line_split[trg_col].split()
                    trg_ids = [trg_dict.get(w, unk_id) for w in trg_words]

                    trg_ids_next = trg_ids + [end_id]
                    trg_ids = [start_id] + trg_ids
                    if len(src_words) + len(trg_words) < len_filter:
                        yield src_ids, trg_ids, trg_ids_next

    return reader


def train(data_file,
          src_dict_size,
          trg_dict_size,
          src_lang="cn",
          src_dict_file=None,
          trg_dict_file=None,
          len_filter=200):

    return reader_creator(data_file, src_lang, src_dict_size, trg_dict_size,
                          src_dict_file, trg_dict_file, len_filter)


test = partial(train, len_filter=100000)


def get_dict(data_file, dict_size, lang, dict_file=None, reverse=False):
    dict_file = "%s_%d.dict" % (lang,
                                dict_size) if dict_file is None else dict_file
    dict_path = os.path.join(DATA_HOME, dict_file)
    assert os.path.exists(dict_path), "Word dictionary does not exist. "
    return __load_dict(data_file, dict_size, lang, dict_file, reverse)
