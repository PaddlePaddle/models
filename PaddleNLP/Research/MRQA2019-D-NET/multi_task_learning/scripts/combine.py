#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module add all train/dev data to a file named "mrqa-combined.raw.json".
"""

import json
import argparse
import glob

# path of train/dev data
parser = argparse.ArgumentParser()
parser.add_argument('path', help='the path of train/dev data')
args = parser.parse_args()
path = args.path

# all train/dev data files
files = glob.glob(path + '/*.raw.json')
print ('files:', files)

# add all train/dev data to "datasets"
with open(files[0]) as fin:
    datasets = json.load(fin)
for i in range(1, len(files)):
    with open(files[i]) as fin:
        dataset = json.load(fin)
    datasets['data'].extend(dataset['data'])

# save to "mrqa-combined.raw.json"
with open(path + '/mrqa-combined.raw.json', 'w') as fout:
    json.dump(datasets, fout, indent=4)
