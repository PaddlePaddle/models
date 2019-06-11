#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import wget
import tarfile

import logging
logger = logging.getLogger(__name__)


__all__ = ['download_weights', 'download_dataset']

WEIGHTS_DIR = os.path.expanduser("~/.paddle/weights")
DATASET_DIR = os.path.expanduser("~/.paddle/dataset")


def download_weights(url):
    """ Download weights from given url. 

    weights saveed in WEIGHTS_DIR.
    if weights specified by url is exists, return weights path
    """
    weights_file = url.split('/')[-1]
    zip_formats = ['.zip', '.tar', '.gz']
    weights_name = weights_file
    for zip_format in zip_formats:
        weights_name = weights_name.replace(zip_format, '')
    weights_path = os.path.join(WEIGHTS_DIR, weights_name)

    if os.path.exists(weights_path):
        logger.info("Found weights in {}".format(weights_path))
    else:
        logger.info("Download weights from {}".format(url))
        if not os.path.exists(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR)
        download_file = os.path.join(WEIGHTS_DIR, weights_file)
        wget.download(url, download_file)
        _decompress(download_file)
        os.remove(download_file)
    
    return weights_path

def download_dataset(url):
    pass

def _decompress(path):
    """
    decompress currently only support tar file
    """
    t = tarfile.open(path)
    t.extractall(path='/'.join(path.split('/')[:-1]))
    t.close()

