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
import requests
import tqdm
import tarfile
import zipfile

import logging
logger = logging.getLogger(__name__)


__all__ = ['get_weights_path', 'get_dataset_path']

WEIGHTS_DIR = os.path.expanduser("~/.paddle/weights")
DATASET_DIR = os.path.expanduser("~/.paddle/dataset")


def get_weights_path(url):
    """Get weights path from WEIGHT_DIR, if not exists,
    download it from url.
    """
    return get_path(url, WEIGHTS_DIR)


def get_dataset_path(path):
    """
    If path exists, return path.
    Otherwise, get dataset path from DATASET_DIR, if not exists,
    download it.
    """
    if os.path.exists(path):
        return path

    urls_map = {
        'coco': ('http://images.cocodataset.org/zips/train2017.zip',
                 'http://images.cocodataset.org/zips/val2017.zip',
                 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',),
        'pascal': ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',)
        }

    logger.info("DATASET_DIR {} not exitst, try searching {} or "
                "downloading dataset...".format(os.path.realpath(path),
                                                DATASET_DIR))
    for dataset, urls in urls_map.items():
        if path.lower().find(dataset) >= 0:
            logger.info("Parse DATASET_DIR {} as dataset "
                        "{}".format(path, dataset))
            data_dir = os.path.join(DATASET_DIR, dataset)
            for url in urls:
                get_path(url, data_dir)
            return data_dir
    
    # not match any dataset in urls_map
    raise ValueError("{} not exists and unknow dataset type".format(path))


def get_path(url, root_dir):
    """ Download from given url to root_dir. 
    if file or directory specified by url is exists under 
    root_dir, return the path directly, otherwise download 
    from url and decompress it, return the path.

    url (str): download url
    root_dir (str): root dir for downloading, it should be 
                    WEIGHTS_DIR or DATASET_DIR
    """
    # parse path after download as decompress under root_dir
    fname = url.split('/')[-1]
    zip_formats = ['.zip', '.tar', '.gz']
    fpath = fname
    for zip_format in zip_formats:
        fpath = fpath.replace(zip_format, '')
    fpath = os.path.join(root_dir, fpath)

    # for decompressed directory name differenct from zip file name
    decompress_name_map = {"VOCtrainval": "VOCdevkit",
                           "annotations_trainval": "annotations"}
    for k, v in decompress_name_map.items():
        if fpath.find(k) >= 0:
            fpath = '/'.join(fpath.split('/')[:-1] + [v])

    if os.path.exists(fpath):
        logger.info("Found {}".format(fpath))
    else:
        _download(url, root_dir)
    
    return fpath


def _download(url, path):
    """
    Download from url, save to path. 

    url (str): download url
    path (str): download to given path
    """
    if not os.path.exists(path):
        os.makedirs(path)

    fname = url.split('/')[-1]
    logger.info("Downloading {} from {}".format(fname, url))

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        raise RuntimeError("Downloading from {} failed with code "
                           "{}!".format(url, req.status_code))

    full_fname = os.path.join(path, fname)
    total_size = req.headers.get('content-length')
    with open(full_fname, 'wb') as f:
        if total_size:
            for chunk in tqdm.tqdm(req.iter_content(chunk_size=1024),
                              total=(int(total_size) + 1023) // 1024,
                              unit='KB'):
                f.write(chunk)
        else:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    _decompress(full_fname)


def _decompress(fname):
    """
    Decompress for zip and tar file
    """
    logger.info("Download finish, decompressing {}...".format(fname))
    fpath='/'.join(fname.split('/')[:-1])
    if fname.find('tar') >= 0:
        with tarfile.open(fname) as tf:
            tf.extractall(path=fpath)
    elif fname.find('zip') >= 0:
        with zipfile.ZipFile(fname) as zf:
            zf.extractall(path=fpath)
    else:
        raise TypeError("Unsupport compress file type {}".format(fname))

    os.remove(fname)

