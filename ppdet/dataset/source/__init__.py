"""
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
"""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
from __future__ import unicode_literals
import os
import copy
from .roidb_source import RoiDbSource
from .simple_source import SimpleSource
from ppdet.utils.download import get_dataset_path


def build_source(config):
    """ build dataset from source data,
        default source type is 'RoiDbSource'
        Args:
          config (dict): should has a structure:
          {
              data_cf (dict):
                  anno_file (str): label file path or image list file path
                  image_dir (str): root dir for images
                  samples (int): samples to load, -1 means all
                  is_shuffle (bool): whether load data in this class
                  load_img (bool): whether load data in this class
                  mixup_epoch (int): parse mixup in first n epoch
                  with_background (bool): whether load background as a class
              cname2cid (dict): the label name to id dictionary
          }
    """
    if 'data_cf' in config:
        data_cf = {k.lower(): v for k, v in config['data_cf'].items()}
    else:
        data_cf = config
    # if DATASET_DIR set and not exists, search dataset under ~/.paddle/dataset
    # if not exists base on DATASET_DIR name (coco or pascal), if not found 
    # under ~/.paddle/dataset, download it.
    if 'dataset_dir' in data_cf:
        dataset_dir = get_dataset_path(data_cf['dataset_dir'])
        if 'anno_file' in data_cf:
            data_cf['anno_file'] = os.path.join(dataset_dir, data_cf['anno_file'])
        data_cf['image_dir'] = os.path.join(dataset_dir, data_cf['image_dir'])
        del data_cf['dataset_dir']
        if data_cf is not config:
            if 'anno_file' in data_cf:
                config['data_cf']['ANNO_FILE'] = os.path.join(dataset_dir,
                                                              data_cf['anno_file'])
            config['data_cf']['IMAGE_DIR'] = os.path.join(dataset_dir,
                                                          data_cf['image_dir'])
    args = copy.deepcopy(data_cf)
    # defaut type is 'RoiDbSource'
    source_type = 'RoiDbSource'
    if 'type' in data_cf:
        if data_cf['type'] in ['VOCSource', 'COCOSource', 'RoiDbSource']:
            source_type = 'RoiDbSource'
        else:
            source_type = data_cf['type']
        del args['type']
    if source_type == 'RoiDbSource':
        return RoiDbSource(**args)
    elif source_type == 'SimpleSource':
        del args['cname2cid']
        for k in ['with_background', 'anno_file']:
            if k in args:
                del args[k]
        return SimpleSource(**args)
    else:
        raise ValueError('not supported source type[%s]' % (source_type))
