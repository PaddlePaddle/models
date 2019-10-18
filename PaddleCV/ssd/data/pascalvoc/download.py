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
import re
import random
import tarfile
import logging

from paddle.dataset.common import download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATASETS = {
    'pascalvoc': [
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
         '6cd6e144f989b92b3379bac3b3de84fd', ),
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
         'c52e279531787c972589f7e41ab4ae64', ),
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
         'b6e924de25625d8de591ea690078ad9f', ),
    ],
}

devkit_dir = './VOCdevkit'
years = ['2007', '2012']


def get_dir(devkit_dir, year, type):
    return osp.join(devkit_dir, 'VOC' + year, type)


def walk_dir(devkit_dir, year):
    filelist_dir = get_dir(devkit_dir, year, 'ImageSets/Main')
    annotation_dir = get_dir(devkit_dir, year, 'Annotations')
    img_dir = get_dir(devkit_dir, year, 'JPEGImages')
    trainval_list = []
    test_list = []
    added = set()

    for _, _, files in os.walk(filelist_dir):
        for fname in files:
            img_ann_list = []
            if re.match('[a-z]+_trainval\.txt', fname):
                img_ann_list = trainval_list
            elif re.match('[a-z]+_test\.txt', fname):
                img_ann_list = test_list
            else:
                continue
            fpath = osp.join(filelist_dir, fname)
            for line in open(fpath):
                name_prefix = line.strip().split()[0]
                if name_prefix in added:
                    continue
                added.add(name_prefix)
                ann_path = osp.join(annotation_dir, name_prefix + '.xml')
                img_path = osp.join(img_dir, name_prefix + '.jpg')
                assert os.path.isfile(ann_path), 'file %s not found.' % ann_path
                assert os.path.isfile(img_path), 'file %s not found.' % img_path
                img_ann_list.append((img_path, ann_path))

    return trainval_list, test_list


def prepare_filelist(devkit_dir, years, output_dir):
    trainval_list = []
    test_list = []
    for year in years:
        trainval, test = walk_dir(devkit_dir, year)
        trainval_list.extend(trainval)
        test_list.extend(test)
    random.shuffle(trainval_list)
    with open(osp.join(output_dir, 'trainval.txt'), 'w') as ftrainval:
        for item in trainval_list:
            ftrainval.write(item[0] + ' ' + item[1] + '\n')

    with open(osp.join(output_dir, 'test.txt'), 'w') as ftest:
        for item in test_list:
            ftest.write(item[0] + ' ' + item[1] + '\n')



def download_decompress_file(data_dir, url, md5):
    logger.info("Downloading from {}".format(url))
    tar_file = download(url, data_dir, md5)
    logger.info("Decompressing {}".format(tar_file))
    with tarfile.open(tar_file) as tf:
        tf.extractall(path=data_dir)
    os.remove(tar_file)


if __name__ == "__main__":
    data_dir = osp.split(osp.realpath(sys.argv[0]))[0]
    for name, infos in DATASETS.items():
        for info in infos:
            download_decompress_file(data_dir, info[0], info[1])
        if name == 'pascalvoc':
            logger.info("create list for pascalvoc dataset.")
            prepare_filelist(devkit_dir, years, data_dir)
        logger.info("Download dataset {} finished.".format(name))
