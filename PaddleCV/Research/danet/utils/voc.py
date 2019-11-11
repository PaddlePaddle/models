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

from utils.base import BaseDataSet


class VOC(BaseDataSet):
    """prepare pascalVOC path_pairs"""
    BASE_DIR = 'VOC2012_SBD'
    NUM_CLASS = 21

    def __init__(self, root='../dataset', split='train', **kwargs):
        super(VOC, self).__init__(root, split, **kwargs)
        if os.sep == '\\':  # windows
            root = root.replace('/', '\\')

        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "please download voc2012 data_set, put in dataset(dir)"
        if split == 'test':
            self.image_path = self._get_cityscapes_pairs(root, split)
        else:
            self.image_path, self.label_path = self._get_cityscapes_pairs(root, split)
        if self.label_path is None:
            pass
        else:
            assert len(self.image_path) == len(self.label_path), "please check image_length = label_length"
        self.print_param() 

    def print_param(self): # 用于核对当前数据集的信息
        if self.label_path is None:
            print('INFO: dataset_root: {}, split: {}, '
                  'base_size: {}, crop_size: {}, scale: {}, '
                  'image_length: {}'.format(self.root, self.split, self.base_size,
                                            self.crop_size, self.scale, len(self.image_path)))
        else:
            print('INFO: dataset_root: {}, split: {}, '
                  'base_size: {}, crop_size: {}, scale: {}, '
                  'image_length: {}, label_length: {}'.format(self.root, self.split, self.base_size,
                                                              self.crop_size, self.scale, len(self.image_path),
                                                              len(self.label_path)))

    @staticmethod
    def _get_cityscapes_pairs(root, split):

        def get_pairs(root, file):
            if file.find('test') == -1:
                file = os.path.join(root, file)
                with open(file, 'r') as f:
                    file_list = f.readlines()
                if os.sep == '\\':  # for windows
                    image_path = [
                        os.path.join(root, 'pascal', 'VOC2012', x.split()[0][1:].replace('/', '\\').replace('\n', ''))
                        for x in file_list]
                    label_path = [os.path.join(root, 'pascal', 'VOC2012', x.split()[1][1:].replace('/', '\\')) for x in
                                  file_list]
                else:
                    image_path = [os.path.join(root, 'pascal', 'VOC2012', x.split()[0][1:]) for x in file_list]
                    label_path = [os.path.join(root, 'pascal', 'VOC2012', x.split()[1][1:]) for x in file_list]
                return image_path, label_path
            else:
                file = os.path.join(root, file)
                with open(file, 'r') as f:
                    file_list = f.readlines()
                if os.sep == '\\':  # for windows
                    image_path = [
                        os.path.join(root, 'pascal', 'VOC2012', x.split()[0][1:].replace('/', '\\').replace('\n', ''))
                        for x in file_list]
                else:
                    image_path = [os.path.join(root, 'pascal', 'VOC2012', x.split()[0][1:]) for x in file_list]
                return image_path

        if split == 'train':
            image_path, label_path = get_pairs(root, 'list/train_aug.txt')
        elif split == 'val':
            image_path, label_path = get_pairs(root, 'list/val.txt')
        elif split == 'test':
            image_path = get_pairs(root, 'list/test.txt')  # 返回文件路径，test_label并不存在
            return image_path
        else:  # 'train_val'
            image_path, label_path = get_pairs(root, 'list/trainval_aug.txt')
        return image_path, label_path

    def get_path_pairs(self):
        return self.image_path, self.label_path
