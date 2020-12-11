# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import collections
import io
import os
import warnings

from paddle.io import Dataset
from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME

from .dataset import TSVDataset

__all__ = ['ChnSentiCorp']


class ChnSentiCorp(TSVDataset):
    """
    ChnSentiCorp (by Tan Songbo at ICT of Chinese Academy of Sciences, and for
    opinion mining)

    """

    URL = "https://bj.bcebos.com/paddlehub-dataset/chnsenticorp.tar.gz"
    MD5 = "fbb3217aeac76a2840d2d5cd19688b07"
    MODE_INFO = collections.namedtuple(
        'MODE_INFO', ('file', 'md5', 'field_indices', 'num_discard_samples'))
    MODES = {
        'train': MODE_INFO(
            os.path.join('chnsenticorp', 'train.tsv'),
            '689360c4a4a9ce8d8719ed500ae80907', (1, 0), 1),
        'dev': MODE_INFO(
            os.path.join('chnsenticorp', 'dev.tsv'),
            '05e4b02561c2a327833e05bbe8156cec', (1, 0), 1),
        'test': MODE_INFO(
            os.path.join('chnsenticorp', 'test.tsv'),
            '917dfc6fbce596bb01a91abaa6c86f9e', (1, 0), 1)
    }

    def __init__(self,
                 mode='train',
                 data_file=None,
                 return_all_fields=False,
                 **kwargs):
        if return_all_fields:
            segments = copy.deepcopy(self.__class__.MODES)
            mode_info = list(modes[mode])
            mode_info[2] = None
            modes[mode] = self.MODE_INFO(*mode_info)
            self.MODES = segments

        self._get_data(data_file, mode, **kwargs)

    def _get_data(self, data_file, mode, **kwargs):
        default_root = DATA_HOME
        filename, data_hash, field_indices, num_discard_samples = self.MODES[
            mode]
        fullname = os.path.join(
            default_root, filename) if data_file is None else os.path.join(
                os.path.expanduser(data_file), filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            if data_file is not None:  # not specified, and no need to warn
                warnings.warn(
                    'md5 check failed for {}, download {} data to {}'.format(
                        filename, self.__class__.__name__, default_root))
            path = get_path_from_url(self.URL, default_root, self.MD5)
            fullname = os.path.join(default_root, filename)
        super(ChnSentiCorp, self).__init__(
            fullname,
            field_indices=field_indices,
            num_discard_samples=num_discard_samples,
            **kwargs)

    def get_labels(self):
        """
        Return labels of the ChnSentiCorp object.
        """
        return ["0", "1"]
