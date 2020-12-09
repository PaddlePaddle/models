import copy
import collections
import os
import warnings

from paddle.dataset.common import DATA_HOME, md5file
from paddle.utils.download import get_path_from_url
import json
from paddle.io import Dataset

__all__ = ['DuReaderYesNo']


class DuReaderYesNo(Dataset):
    SEGMENT_INFO = collections.namedtuple('SEGMENT_INFO', ('file', 'md5'))

    DATA_URL = 'https://dataset-bj.cdn.bcebos.com/qianyan/dureader_yesno-data.tar.gz'

    SEGMENTS = {
        'train': SEGMENT_INFO(
            os.path.join('dureader_yesno-data', 'train.json'),
            'dc2dac669a113866a6480a0b10cd50bf'),
        'dev': SEGMENT_INFO(
            os.path.join('dureader_yesno-data', 'dev.json'),
            '185958e46ba556b38c6a7cc63f3a2135'),
        'test': SEGMENT_INFO(
            os.path.join('dureader_yesno-data', 'test.json'),
            '185958e46ba556b38c6a7cc63f3a2135')
    }

    def __init__(self, segment='train', root=None, **kwargs):

        self._get_data(root, segment, **kwargs)
        self._transform_func = None

        if segment == 'train':
            self.is_training = True
        else:
            self.is_training = False

        self._read()

    def _get_data(self, root, segment, **kwargs):
        default_root = os.path.join(DATA_HOME, 'DuReader')

        filename, data_hash = self.SEGMENTS[segment]

        fullname = os.path.join(default_root,
                                filename) if root is None else os.path.join(
                                    os.path.expanduser(root), filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            if root is not None:  # not specified, and no need to warn
                warnings.warn(
                    'md5 check failed for {}, download {} data to {}'.format(
                        filename, self.__class__.__name__, default_root))

            get_path_from_url(self.DATA_URL, default_root)

        self.full_path = fullname

    def _read(self):
        data_lines = []
        with open(self.full_path, "r", encoding="utf8") as reader:
            data_lines += reader.readlines()

        examples = []
        for entry in data_lines:
            source = json.loads(entry.strip())
            examples.append(
                [source['question'], source['answer'], source['yesno_answer']])

        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def get_labels(self):
        """
        Return labels of the GlueCoLA object.
        """
        return ["Yes", "No", "Depends"]
