import tarfile
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
        src_dict = __to_dict__(f.extractfile(names[0]), dict_size)
    return src_dict


def reader_creator(tar_file, file_name, dict_size):
    def reader():
        src_dict = __read_to_dict__(tar_file, dict_size)
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
                    src_seq = line_split[0]  # one source sequence
                    src_words = src_seq.split()
                    src_ids = [
                        src_dict.get(w, UNK_IDX)
                        for w in [START] + src_words + [END]
                    ]
                    trg_seq = line_split[1]  # one target sequence
                    trg_words = trg_seq.split()
                    trg_ids = [src_dict.get(w, UNK_IDX) for w in trg_words]

                    # remove sequence whose length > 80 in training mode
                    if len(src_ids) > 80 or len(trg_ids) > 80:
                        continue
                    trg_ids_next = trg_ids + [src_dict[END]]
                    trg_ids = [src_dict[START]] + trg_ids

                    yield src_ids, trg_ids, trg_ids_next

    return reader


def train(dict_size):
    return reader_creator('train_data.tar.gz', 'train', dict_size)


def test(dict_size):
    return reader_creator('train_data.tar.gz', 'test', dict_size)


def get_dict(dict_size, reverse=True):
    src_dict = __read_to_dict__(tar_file, dict_size)
    return src_dict
