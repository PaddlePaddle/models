# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import re
import random
import shutil

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
                new_ann_path = osp.join('./VOCdevkit/VOC_all/Annotations/',
                                        name_prefix + '.xml')
                new_img_path = osp.join('./VOCdevkit/VOC_all/JPEGImages/',
                                        name_prefix + '.jpg')
                shutil.copy(ann_path, new_ann_path)
                shutil.copy(img_path, new_img_path)
                img_ann_list.append(name_prefix)

    return trainval_list, test_list


def prepare_filelist(devkit_dir, years, output_dir):
    os.makedirs('./VOCdevkit/VOC_all/Annotations/')
    os.makedirs('./VOCdevkit/VOC_all/ImageSets/Main/')
    os.makedirs('./VOCdevkit/VOC_all/JPEGImages/')
    trainval_list = []
    test_list = []
    for year in years:
        trainval, test = walk_dir(devkit_dir, year)
        trainval_list.extend(trainval)
        test_list.extend(test)
    random.shuffle(trainval_list)
    with open(osp.join(output_dir, 'train.txt'), 'w') as ftrainval:
        for item in trainval_list:
            ftrainval.write(item + '\n')

    with open(osp.join(output_dir, 'val.txt'), 'w') as fval:
        with open(osp.join(output_dir, 'test.txt'), 'w') as ftest:
            ct = 0
            for item in test_list:
                ct += 1
                fval.write(item + '\n')
                if ct <= 1000:
                    ftest.write(item + '\n')


if __name__ == '__main__':
    prepare_filelist(devkit_dir, years, '.')
