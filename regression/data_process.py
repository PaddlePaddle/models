import tarfile
import gzip
from paddle.v2.parameters import Parameters

__all__ = ['train', 'test', 'build_dict']

START = "<s>"
END = "<e>"
UNK = "<unk>"
UNK_IDX = 2


def __read_to_dict__(tar_file, dict_size):
    def __to_dict__(fd, size):
        out_dict = dict()
        for line_count, line in enumerate(fd):
            if line_count < size:
                out_dict[line.strip()] = line_count
            else:
                break
        return out_dict

    with tarfile.open(tar_file, mode='r') as f:
        names = [
            each_item.name for each_item in f if each_item.name.endswith("dict")
        ]
        assert len(names) == 1
        word_dict = __to_dict__(f.extractfile(names[0]), dict_size)
    return word_dict


def reader_creator(tar_file, file_name, dict_size):
    def reader():
        word_dict = __read_to_dict__(tar_file, dict_size)
        with tarfile.open(tar_file, mode='r') as f:
            names = [
                each_item.name for each_item in f
                if each_item.name.endswith(file_name)
            ]
            for name in names:
                for line in f.extractfile(name):
                    line_split = line.strip().split('\t')
                    if len(line_split) != 2:
                        continue
                    first_seq = line_split[0]  # first sequence
                    first_words = first_seq.split()
                    first_ids = [
                        word_dict.get(w, UNK_IDX)
                        for w in [START] + first_words + [END]
                    ]
                    second_seq = line_split[
                        1]  # second sequence relate to first
                    second_words = second_seq.split()
                    second_ids = [
                        word_dict.get(w, UNK_IDX) for w in second_words
                    ]

                    # remove sequence whose length > 80 in training mode
                    if len(first_ids) > 80 or len(second_ids) > 80:
                        continue
                    second_ids_next = second_ids + [word_dict[END]]
                    second_ids = [word_dict[START]] + second_ids

                    yield first_ids, second_ids, second_ids_next

    return reader


def train(dict_size):
    return reader_creator('train_data.tar.gz', 'train', dict_size)


def test(dict_size):
    return reader_creator('train_data.tar.gz', 'test', dict_size)


def get_dict(dict_size, reverse=True):
    word_dict = __read_to_dict__(tar_file, dict_size)
    return word_dict
