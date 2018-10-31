#  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License. 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

LAYER_TYPES = [
        "net",
        "convolutional",
        "shortcut",
        "route",
        "upsample",
        "maxpool",
        "yolo",
        ]

class ConfigPaser(object):
    def __init__(self, config_path):
        self.config_path = config_path
    
    def parse(self):
        with open(self.config_path) as cfg_file:
            model_defs = []
            for line in cfg_file.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                if line.startswith('#'):
                    continue
                if line.startswith('['):
                    layer_type = line[1:-1].strip()
                    if layer_type not in LAYER_TYPES:
                        print("Unknow config layer type: ", layer_type)
                        return None
                    model_defs.append({})
                    model_defs[-1]['type'] = layer_type
                else:
                    key, value = line.split('=')
                    model_defs[-1][key.strip()] = value.strip()

        return model_defs


