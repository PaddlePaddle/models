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

# function:
#   tool used to convert roidb data in json to pickled file

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import xml.etree.ElementTree as ET


def parse_args():
    """ parse arguments
    """
    parser = argparse.ArgumentParser(
        description='Generate Dataset\'s VOC label list.')

    parser.add_argument('-list', action='append',
                        dest='db_list', default=[],
                        help='The datset path list')

    args = parser.parse_args()
    return args


def save_label_list(args):
    path_list = args.db_list
    category_list = []
    for path in path_list:
        train_txt_path = os.path.join(path, 'ImageSets',
                                      'Main', 'train.txt')
        annot_path = os.path.join(path, 'Annotations')
        xml_list = []
        with open(train_txt_path, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                xml_list.append(line.strip()+'.xml')
        for file_name in xml_list:
            xml_file_path = os.path.join(annot_path, file_name)
            if not os.path.isfile(xml_file_path):
                continue
            tree = ET.parse(xml_file_path)
            objs = tree.findall('object')
            for obj in objs:
                cat = obj.find('name').text
                if cat not in category_list:
                    category_list.append(cat)
    for path in path_list:
        with open(os.path.join(path, 'label_list.txt'), 'w') as fw:
            for category in category_list:
                fw.write(category+'\n')


if __name__ == "__main__":
    """ make sure your data is stored in 'data/${args.dataset}'

    usage:
        python generate_voc_label.py -list 'voc2014' -list 'voc2017'
    """
    args = parse_args()
    save_label_list(args)
