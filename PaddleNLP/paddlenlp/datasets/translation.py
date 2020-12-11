import os
import io

from functools import partial
import numpy as np

import paddle
from paddle.utils.download import get_path_from_url
from paddlenlp.data import Vocab, Pad
from paddlenlp.data.sampler import SamplerHelper

DATA_HOME = "/root/.paddlenlp/datasets"

__all__ = ['TranslationDataset', 'IWSLT15']


def read_raw_files(corpus_path):
    """Read raw files, return raw data"""
    data = []
    (f_mode, f_encoding, endl) = ("r", "utf-8", "\n")
    with io.open(corpus_path, f_mode, encoding=f_encoding) as f_corpus:
        for line in f_corpus.readlines():
            data.append(line.strip())
    return data


def get_raw_data(data_dir, train_filenames, valid_filenames, test_filenames,
                 data_select):
    data_dict = {}
    file_select = {
        'train': train_filenames,
        'dev': valid_filenames,
        'test': test_filenames
    }
    for mode in data_select:
        src_filename, tgt_filename = file_select[mode]
        src_path = os.path.join(data_dir, src_filename)
        tgt_path = os.path.join(data_dir, tgt_filename)
        src_data = read_raw_files(src_path)
        tgt_data = read_raw_files(tgt_path)

        data_dict[mode] = [(src_data[i], tgt_data[i])
                           for i in range(len(src_data))]
    return data_dict


def setup_datasets(train_filenames,
                   valid_filenames,
                   test_filenames,
                   data_select,
                   root=None):
    # Input check
    target_select = ('train', 'dev', 'test')
    if isinstance(data_select, str):
        data_select = (data_select, )
    if not set(data_select).issubset(set(target_select)):
        raise TypeError(
            'A subset of data selection {} is supported but {} is passed in'.
            format(target_select, data_select))

    raw_data = get_raw_data(root, train_filenames, valid_filenames,
                            test_filenames, data_select)

    datasets = []
    for mode in data_select:
        datasets.append(TranslationDataset(raw_data[mode]))
    return tuple(datasets)


def vocab_func(vocab, unk_token):
    def func(tok_iter):
        return [
            vocab[tok] if tok in vocab else vocab[unk_token] for tok in tok_iter
        ]

    return func


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def get_default_tokenizer():
    """Only support split tokenizer
    """

    def _split_tokenizer(x):
        return x.split()

    return _split_tokenizer


class TranslationDataset(paddle.io.Dataset):
    """
    TranslationDataset, provide tuple (source and target) raw data.
    
    Args:
        data(list): Raw data. It is a list of tuple or list, each sample of
            data contains two element, source and target.
    """
    URL = None
    train_filenames = (None, None)
    valid_filenames = (None, None)
    test_filenames = (None, None)
    src_vocab_filename = None
    tgt_vocab_filename = None
    dataset_dirname = None

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_data(cls, root=None):
        """
        Download dataset if any data file doesn't exist.
        Args:
            root (str, optional): data directory to save dataset. If not
                provided, dataset will be saved in
                `/root/.paddlenlp/datasets/machine_translation`. Default: None.
        Returns:
            str: All file paths of dataset.

        Examples:
            .. code-block:: python
                from paddlenlp.datasets import IWSLT15
                data_path = IWSLT15.get_data()
        """
        if root is None:
            root = os.path.join(DATA_HOME, 'machine_translation')
            data_dir = os.path.join(root, cls.dataset_dirname)
        if not os.path.exists(root):
            os.makedirs(root)
            print("IWSLT will be downloaded at ", root)
            get_path_from_url(cls.URL, root)
            print("Downloaded success......")
        else:
            filename_list = [
                cls.train_filenames[0], cls.train_filenames[1],
                cls.valid_filenames[0], cls.valid_filenames[0],
                cls.src_vocab_filename, cls.tgt_vocab_filename
            ]
            for filename in filename_list:
                file_path = os.path.join(data_dir, filename)
                if not os.path.exists(file_path):
                    print(
                        "The dataset is incomplete and will be re-downloaded.")
                    get_path_from_url(cls.URL, root)
                    print("Downloaded success......")
                    break
        return data_dir

    @classmethod
    def get_vocab(cls, root=None):
        """
        Load vocab from vocab files. It vocab files don't exist, the will
        be downloaded.

        Args:
            root (str, optional): Data directory to save dataset. If not provided,
                dataset will be save in `/root/.paddlenlp/datasets/machine_translation`.
                If vocab files exist, they won't be overwritten. Default: None.
        Returns:
            tuple: Source vocab and target vocab.

        Examples:
            .. code-block:: python
                from paddlenlp.datasets import IWSLT15
                (src_vocab, tgt_vocab) = IWSLT15.get_vocab()

        """
        data_path = cls.get_data(root)

        # Get vocab_func
        src_file_path = os.path.join(data_path, cls.src_vocab_filename)
        tgt_file_path = os.path.join(data_path, cls.tgt_vocab_filename)

        src_vocab = Vocab.load_vocabulary(src_file_path, cls.unk_token,
                                          cls.bos_token, cls.eos_token)

        tgt_vocab = Vocab.load_vocabulary(tgt_file_path, cls.unk_token,
                                          cls.bos_token, cls.eos_token)
        return (src_vocab, tgt_vocab)

    @classmethod
    def get_default_transform_func(cls, root=None):
        """Get default transform function, which transforms raw data to id.
        Args:
            root(str, optional): Data directory of dataset.
        Returns:
            tuple: Two transform functions, for source and target data. 
        Examples:
            .. code-block:: python
                from paddlenlp.datasets import IWSLT15
                transform_func = IWSLT15.get_default_transform_func()
        """
        # Get default tokenizer
        src_tokenizer = get_default_tokenizer()
        tgt_tokenizer = get_default_tokenizer()
        src_text_vocab_transform = sequential_transforms(src_tokenizer)
        tgt_text_vocab_transform = sequential_transforms(tgt_tokenizer)

        (src_vocab, tgt_vocab) = cls.get_vocab(root)
        src_text_transform = sequential_transforms(
            src_text_vocab_transform, vocab_func(src_vocab, cls.unk_token))
        tgt_text_transform = sequential_transforms(
            tgt_text_vocab_transform, vocab_func(tgt_vocab, cls.unk_token))
        return (src_text_transform, tgt_text_transform)


class IWSLT15(TranslationDataset):
    """
    IWSLT15 Vietnames to English translation dataset.

    Args:
        data(list|optional): Raw data. It is a list of tuple, each tuple
            consists of source and target data. Default: None.
        vocab(tuple|optional): Tuple of Vocab object or dict. It consists of
            source and target language vocab. Default: None.
    Examples:
        .. code-block:: python
            from paddlenlp.datasets import IWSLT15
            train_dataset = IWSLT15('train')
            train_dataset, valid_dataset = IWSLT15.get_datasets(["train", "dev"])

    """
    URL = "https://paddlenlp.bj.bcebos.com/datasets/iwslt15.en-vi.tar.gz"
    train_filenames = ("train.en", "train.vi")
    valid_filenames = ("tst2012.en", "tst2012.vi")
    test_filenames = ("tst2013.en", "tst2013.vi")
    src_vocab_filename = "vocab.en"
    tgt_vocab_filename = "vocab.vi"
    unk_token = '<unk>'
    bos_token = '<s>'
    eos_token = '</s>'
    dataset_dirname = "iwslt15.en-vi"

    def __init__(self, mode='train', root=None, transform_func=None):
        # Input check
        segment_select = ('train', 'dev', 'test')
        if mode not in segment_select:
            raise TypeError(
                '`train`, `dev` or `test` is supported but `{}` is passed in'.
                format(mode))
        if transform_func is not None:
            if len(transform_func) != 2:
                raise ValueError("`transform_func` must have length of two for"
                                 "source and target")
        # Download data
        data_path = IWSLT15.get_data(root)
        dataset = setup_datasets(self.train_filenames, self.valid_filenames,
                                 self.test_filenames, [mode], data_path)[0]
        self.data = dataset.data
        if transform_func is not None:
            self.data = [(transform_func[0](data[0]),
                          transform_func[1](data[1])) for data in self.data]


# For test, not API
def prepare_train_input(insts, pad_id):
    src, src_length = Pad(pad_val=pad_id, ret_length=True)(
        [inst[0] for inst in insts])
    tgt, tgt_length = Pad(pad_val=pad_id, ret_length=True)(
        [inst[1] for inst in insts])
    return src, src_length, tgt[:, :-1], tgt[:, 1:, np.newaxis]


if __name__ == '__main__':
    batch_size = 32
    pad_id = 2

    transform_func = IWSLT15.get_default_transform_func()
    train_dataset = IWSLT15(transform_func=transform_func)

    key = (lambda x, data_source: len(data_source[x][0]))
    train_batch_sampler = SamplerHelper(train_dataset).shuffle().sort(
        key=key, buffer_size=batch_size * 20).batch(
            batch_size=batch_size, drop_last=True).shard()

    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=partial(
            prepare_train_input, pad_id=pad_id))

    for data in train_loader:
        print(data)
        break
