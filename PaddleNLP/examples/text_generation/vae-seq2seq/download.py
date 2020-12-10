# -*- coding: utf-8 -*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Script for downloading training data.
'''
import os
import sys
import shutil
import argparse
import urllib
import tarfile
import urllib.request
import zipfile

URLLIB = urllib.request

TASKS = ['ptb', 'yahoo']
TASK2PATH = {
    'ptb': 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz',
    'yahoo':
    'https://drive.google.com/file/d/13IsiffVjcQ-wrrbBGMwiG3sYf-DFxtXH/view?usp=sharing/yahoo.zip',
}


def un_tar(tar_name, dir_name):
    try:
        t = tarfile.open(tar_name)
        t.extractall(path=dir_name)
        return True
    except Exception as e:
        print(e)
        return False


def un_zip(filepath, dir_name):
    z = zipfile.ZipFile(filepath, 'r')
    for file in z.namelist():
        z.extract(file, dir_name)


def download_ptb_and_extract(task, data_path):
    print('Downloading and extracting %s...' % task)
    data_file = os.path.join(data_path, TASK2PATH[task].split('/')[-1])
    URLLIB.urlretrieve(TASK2PATH[task], data_file)
    un_tar(data_file, data_path)
    os.remove(data_file)
    src_dir = os.path.join(data_path, 'simple-examples')
    dst_dir = os.path.join(data_path, 'ptb')
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    shutil.copy(os.path.join(src_dir, 'data/ptb.train.txt'), dst_dir)
    shutil.copy(os.path.join(src_dir, 'data/ptb.valid.txt'), dst_dir)
    shutil.copy(os.path.join(src_dir, 'data/ptb.test.txt'), dst_dir)
    print('\tCompleted!')


def download_yahoo_dataset(task, data_dir):
    url = TASK2PATH[task]
    # id is between `/d/` and '/'
    url_suffix = url[url.find('/d/') + 3:]
    if url_suffix.find('/') == -1:
        # if there's no trailing '/'
        file_id = url_suffix
    else:
        file_id = url_suffix[:url_suffix.find('/')]

    try:
        import requests
    except ImportError:
        print("The requests library must be installed to download files from "
              "Google drive. Please see: https://github.com/psf/requests")
        raise

    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    gurl = "https://docs.google.com/uc?export=download"
    sess = requests.Session()
    response = sess.get(gurl, params={'id': file_id}, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        params = {'id': file_id, 'confirm': token}
        response = sess.get(gurl, params=params, stream=True)

    filename = 'yahoo.zip'
    filepath = os.path.join(data_dir, filename)
    CHUNK_SIZE = 32768
    with open(filepath, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    un_zip(filepath, data_dir)
    os.remove(filepath)

    print('Successfully downloaded yahoo')


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_dir',
        help='directory to save data to',
        type=str,
        default='data')
    parser.add_argument(
        '-t',
        '--task',
        help='tasks to download data for as a comma separated string',
        type=str,
        default='ptb')
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    if args.task == 'yahoo':
        if args.data_dir == 'data':
            args.data_dir = './'
        download_yahoo_dataset(args.task, args.data_dir)
    else:
        download_ptb_and_extract(args.task, args.data_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
