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
import shutil
import requests
import tqdm
import hashlib
import tarfile
import zipfile

import logging
logger = logging.getLogger(__name__)


__all__ = ['get_weights_path', 'get_dataset_path']

WEIGHTS_HOME = os.path.expanduser("~/.cache/paddle/weights")
DATASET_HOME = os.path.expanduser("~/.cache/paddle/dataset")

DATASETS = {
    'coco': [
        ('http://images.cocodataset.org/zips/train2017.zip',
         'cced6f7f71b7629ddf16f17bbcfab6b2',),
        ('http://images.cocodataset.org/zips/val2017.zip',
         '442b8da7639aecaf257c1dceb8ba8c80',),
        ('http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
         'f4bbac642086de4f52a3fdda2de5fa2c',),
    ],
    'pascal': [
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
         '6cd6e144f989b92b3379bac3b3de84fd',)
    ],
}

DOWNLOAD_RETRY_LIMIT = 3


def get_weights_path(url):
    """Get weights path from WEIGHT_HOME, if not exists,
    download it from url.
    """
    return get_path(url, WEIGHTS_HOME)


def get_dataset_path(path):
    """
    If path exists, return path.
    Otherwise, get dataset path from DATASET_HOME, if not exists,
    download it.
    """
    if os.path.exists(path):
        logger.debug("Data path: {}".format(os.path.realpath(path)))
        return path

    logger.info("DATASET_DIR {} not exitst, try searching {} or "
                "downloading dataset...".format(os.path.realpath(path),
                                                DATASET_HOME))
    for name, dataset in DATASETS.items():
        if path.lower().find(name) >= 0:
            logger.info("Parse DATASET_DIR {} as dataset "
                        "{}".format(path, name))
            data_dir = os.path.join(DATASET_HOME, name)
            for url, md5sum in dataset:
                get_path(url, data_dir, md5sum)
            return data_dir
    
    # not match any dataset in DATASETS
    raise ValueError("{} not exists and unknow dataset type".format(path))


def get_path(url, root_dir, md5sum=None):
    """ Download from given url to root_dir. 
    if file or directory specified by url is exists under 
    root_dir, return the path directly, otherwise download 
    from url and decompress it, return the path.

    url (str): download url
    root_dir (str): root dir for downloading, it should be 
                    WEIGHTS_HOME or DATASET_HOME
    md5sum (str): md5 sum of download package
    """
    # parse path after download as decompress under root_dir
    fname = url.split('/')[-1]
    zip_formats = ['.zip', '.tar', '.gz']
    fpath = fname
    for zip_format in zip_formats:
        fpath = fpath.replace(zip_format, '')
    fullpath = os.path.join(root_dir, fpath)

    # for decompressed directory name differenct from zip file name
    decompress_name_map = {"VOCtrainval": "VOCdevkit",
                           "annotations_trainval": "annotations"}
    for k, v in decompress_name_map.items():
        if fullpath.find(k) >= 0:
            fullpath = '/'.join(fullpath.split('/')[:-1] + [v])

    if os.path.exists(fullpath):
        logger.info("Found {}".format(fullpath))
    else:
        _download(url, root_dir, md5sum)
    
    return fullpath


def _download(url, path, md5sum=None):
    """
    Download from url, save to path. 

    url (str): download url
    path (str): download to given path
    """
    if not os.path.exists(path):
        os.makedirs(path)

    fname = url.split('/')[-1]
    fullname = os.path.join(path, fname)
    retry_cnt = 0
    while not (os.path.exists(fullname) and _md5check(fullname, md5sum)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError("Download from {} failed.".format(url))

        logger.info("Downloading {} from {}".format(fname, url))

        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code "
                               "{}!".format(url, req.status_code))

        total_size = req.headers.get('content-length')
        with open(fullname, 'wb') as f:
            if total_size:
                for chunk in tqdm.tqdm(req.iter_content(chunk_size=1024),
                                  total=(int(total_size) + 1023) // 1024,
                                  unit='KB'):
                    f.write(chunk)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

    _decompress(fullname)


def _md5check(fullname, md5sum=None):
    if md5sum is None:
        return True

    logger.info("File {} md5 checking...".format(fullname))
    md5 = hashlib.md5()
    with open(fullname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        logger.info("File {} md5 check failed, {}(calc) != "
                    "{}(base)".format(fullname, calc_md5sum, md5sum))
        return False
    return True


def _decompress(fname):
    """
    Decompress for zip and tar file
    """
    logger.info("Decompressing {}...".format(fname))

    # For protecting decompressing interupted, 
    # decompress to fpath_tmp directory firstly, if decompress 
    # successed, move decompress files to fpath and delete 
    # fpath_tmp and download file.
    fpath = '/'.join(fname.split('/')[:-1])
    fpath_tmp = os.path.join(fpath, 'tmp')
    if os.path.isdir(fpath_tmp):
        shutil.rmtree(fpath_tmp)
        os.makedirs(fpath_tmp)

    if fname.find('tar') >= 0:
        with tarfile.open(fname) as tf:
            tf.extractall(path=fpath_tmp)
    elif fname.find('zip') >= 0:
        with zipfile.ZipFile(fname) as zf:
            zf.extractall(path=fpath_tmp)
    else:
        raise TypeError("Unsupport compress file type {}".format(fname))

    for f in os.listdir(fpath_tmp):
        shutil.move(os.path.join(fpath_tmp, f), os.path.join(fpath, f))
    os.rmdir(fpath_tmp)
    os.remove(fname)

