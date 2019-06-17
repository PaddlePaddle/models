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

__all__ = [
    'Detectors',
    'Detectors',
    'RPNHeads',
    'RoIExtractors',
    'BBoxHeadConvs',
    'BBoxHeads',
    'SSDHeads',
    'RetinaHeads',
    'Necks',
]


class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, name):
        if name not in self._module_dict:
            raise KeyError('{} is not found in {}. You should '
                           'regist it at first.'.format(name, self.name))
        return self._module_dict[name]

    def register(self, cls):
        module_name = cls.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(module_name,
                                                                   self.name))
        self._module_dict[module_name] = cls
        return cls


Backbones = Registry('Backbones')
Detectors = Registry('Detectors')

RPNHeads = Registry('RPNHead')
RoIExtractors = Registry('RoIExtractor')
BBoxHeadConvs = Registry('BBoxHeadConv')
BBoxHeads = Registry('BBoxHead')
MaskHeads = Registry('MaskHead')
RetinaHeads = Registry('RetinaHead')

YOLOHeads = Registry('YOLOHeads')
SSDHeads = Registry('SSDHead')
Necks = Registry('Necks')
