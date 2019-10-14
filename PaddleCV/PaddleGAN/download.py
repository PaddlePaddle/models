#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import print_function
from PIL import Image

import numpy as np
import os
import sys
import gzip
import argparse
import requests
import six
import hashlib
import zipfile

parser = argparse.ArgumentParser(description='Download dataset.')
#TODO  add celeA dataset
parser.add_argument(
    '--dataset',
    type=str,
    default='mnist',
    help='name of dataset to download [mnist]')


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def download_mnist(dir_path):
    URL_DIC = {}
    URL_PREFIX = 'http://yann.lecun.com/exdb/mnist/'
    TEST_IMAGE_URL = URL_PREFIX + 't10k-images-idx3-ubyte.gz'
    TEST_IMAGE_MD5 = '9fb629c4189551a2d022fa330f9573f3'
    TEST_LABEL_URL = URL_PREFIX + 't10k-labels-idx1-ubyte.gz'
    TEST_LABEL_MD5 = 'ec29112dd5afa0611ce80d1b7f02629c'
    TRAIN_IMAGE_URL = URL_PREFIX + 'train-images-idx3-ubyte.gz'
    TRAIN_IMAGE_MD5 = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'
    TRAIN_LABEL_URL = URL_PREFIX + 'train-labels-idx1-ubyte.gz'
    TRAIN_LABEL_MD5 = 'd53e105ee54ea40749a09fcbcd1e9432'
    URL_DIC[TRAIN_IMAGE_URL] = TRAIN_IMAGE_MD5
    URL_DIC[TRAIN_LABEL_URL] = TRAIN_LABEL_MD5
    URL_DIC[TEST_IMAGE_URL] = TEST_IMAGE_MD5
    URL_DIC[TEST_LABEL_URL] = TEST_LABEL_MD5

    ###    print(url)
    for url in URL_DIC:
        md5sum = URL_DIC[url]

        data_dir = os.path.join(dir_path, 'mnist')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        filename = os.path.join(data_dir, url.split('/')[-1])
        retry = 0
        retry_limit = 3
        while not (os.path.exists(filename) and md5file(filename) == md5sum):
            if os.path.exists(filename):
                sys.stderr.write("file %s  md5 %s" %
                                 (md5file(filename), md5sum))
            if retry < retry_limit:
                retry += 1
            else:
                raise RuntimeError("Cannot download {0} within retry limit {1}".
                                   format(url, retry_limit))
            sys.stderr.write("Cache file %s not found, downloading %s" %
                             (filename, url))
            r = requests.get(url, stream=True)
            total_length = r.headers.get('content-length')

            if total_length is None:
                with open(filename, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            else:
                with open(filename, 'wb') as f:
                    dl = 0
                    total_length = int(total_length)
                    for data in r.iter_content(chunk_size=4096):
                        if six.PY2:
                            data = six.b(data)
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        sys.stderr.write("\r[%s%s]" % ('=' * done,
                                                       ' ' * (50 - done)))
                        sys.stdout.flush()
        sys.stderr.write("\n")
        sys.stdout.flush()
        print(filename)


def download_cycle_pix(dir_path, dataname):
    URL_PREFIX = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/'
    IMAGE_URL = '{}.zip'.format(dataname)
    url = URL_PREFIX + IMAGE_URL
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    r = requests.get(url, stream=True)
    total_length = float(r.headers.get('content-length'))
    filename = os.path.join(dir_path, IMAGE_URL)
    print(filename)
    if not os.path.exists(filename):
        dl = 0
        with open(filename, "wb") as f:
            for data in r.iter_content(chunk_size=4096):
                if six.PY2:
                    data = six.b(data)
                dl += len(data)
                f.write(data)
                done = int(100 * dl / total_length)
                sys.stderr.write("\r[{}{}] {}% ".format('=' * done, ' ' * (
                    100 - done), done))
                sys.stdout.flush()
    else:
        sys.stderr.write('{}.zip is EXIST, DO NOT NEED to download it again.'.
                         format(dataname))
    ### unzip .zip file
    if not os.path.exists(os.path.join(dir_path, '{}'.format(dataname))):
        zip_f = zipfile.ZipFile(filename, 'r')
        for zip_file in zip_f.namelist():
            zip_f.extract(zip_file, dir_path)

    ### generator .txt file according to dirs
    dirs = os.listdir(os.path.join(dir_path, '{}'.format(dataname)))
    for d in dirs:
        txt_file = d + '.txt'
        txt_dir = os.path.join(dir_path, dataname)
        f = open(os.path.join(txt_dir, txt_file), 'w')
        for fil in os.listdir(os.path.join(txt_dir, d)):
            wl = d + '/' + fil + '\n'
            f.write(wl)
        f.close()
    sys.stderr.write("\n")


if __name__ == '__main__':
    args = parser.parse_args()
    cycle_pix_dataset = [
        'apple2orange', 'summer2winter_yosemite', 'horse2zebra', 'monet2photo',
        'cezanne2photo', 'ukiyoe2photo', 'vangogh2photo', 'maps', 'facades',
        'iphone2dslr_flower', 'ae_photos', 'mini'
    ]

    pwd = os.path.join(os.path.dirname(__file__), 'data')
    if args.dataset == 'mnist':
        print('Download dataset: {}'.format(args.dataset))
        download_mnist(pwd)
    elif args.dataset in cycle_pix_dataset:
        print('Download dataset: {}'.format(args.dataset))
        download_cycle_pix(pwd, args.dataset)
    else:
        print('Please download by yourself, thanks')
