import copy
import collections
import json
import os
import warnings

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddle.io import Dataset
from paddlenlp.utils.env import DATA_HOME
from .squad import SQuAD

__all__ = ['DuReaderRobust']


class DuReaderRobust(SQuAD):
    SEGMENT_INFO = collections.namedtuple('SEGMENT_INFO', ('file', 'md5'))

    DATA_URL = 'https://dataset-bj.cdn.bcebos.com/qianyan/dureader_robust-data.tar.gz'

    SEGMENTS = {
        'train': SEGMENT_INFO(
            os.path.join('dureader_robust-data', 'train.json'),
            'dc2dac669a113866a6480a0b10cd50bf'),
        'dev': SEGMENT_INFO(
            os.path.join('dureader_robust-data', 'dev.json'),
            '185958e46ba556b38c6a7cc63f3a2135'),
        'test': SEGMENT_INFO(
            os.path.join('dureader_robust-data', 'test.json'),
            '185958e46ba556b38c6a7cc63f3a2135')
    }

    def __init__(self,
                 tokenizer,
                 segment='train',
                 version_2_with_negative=True,
                 root=None,
                 doc_stride=128,
                 max_query_length=64,
                 max_seq_length=512,
                 **kwargs):

        super(DuReaderRobust, self).__init__(
            tokenizer=tokenizer,
            segment=segment,
            version_2_with_negative=False,
            root=root,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            max_seq_length=max_seq_length,
            **kwargs)

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
