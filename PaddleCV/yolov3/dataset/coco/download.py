# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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

import os
import os.path as osp
import sys
import zipfile
import logging

from paddle.dataset.common import download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATASETS = {
    'coco': [
        # coco2017
        ('http://images.cocodataset.org/zips/train2017.zip',
         'cced6f7f71b7629ddf16f17bbcfab6b2', ),
        ('http://images.cocodataset.org/zips/val2017.zip',
         '442b8da7639aecaf257c1dceb8ba8c80', ),
        ('http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
         'f4bbac642086de4f52a3fdda2de5fa2c', ),
        # coco2014
        ('http://images.cocodataset.org/zips/train2014.zip',
         '0da8c0bd3d6becc4dcb32757491aca88', ),
        ('http://images.cocodataset.org/zips/val2014.zip',
         'a3d79f5ed8d289b7a7554ce06a5782b3', ),
        ('http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
         '0a379cfc70b0e71301e0f377548639bd', ),
    ],
}


def download_decompress_file(data_dir, url, md5):
    logger.info("Downloading from {}".format(url))
    zip_file = download(url, data_dir, md5)
    logger.info("Decompressing {}".format(zip_file))
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(path=data_dir)
    os.remove(zip_file)


if __name__ == "__main__":
    data_dir = osp.split(osp.realpath(sys.argv[0]))[0]
    for name, infos in DATASETS.items():
        for info in infos:
            download_decompress_file(data_dir, info[0], info[1])
        logger.info("Download dataset {} finished.".format(name))
