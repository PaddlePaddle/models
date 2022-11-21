# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import glob
import setuptools
from setuptools import setup, find_packages

VERSION = '0.1.0'

with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()


def readme():
    with open('./README.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README


def get_package_data_files(package, data, package_dir=None):
    """
    Helps to list all specified files in package including files in directories
    since `package_data` ignores directories.
    """
    if package_dir is None:
        package_dir = os.path.join(*package.split('.'))
    all_files = []
    for f in data:
        path = os.path.join(package_dir, f)
        if os.path.isfile(path):
            all_files.append(f)
            continue
        for root, _dirs, files in os.walk(path, followlinks=True):
            root = os.path.relpath(root, package_dir)
            for file in files:
                file = os.path.join(root, file)
                if file not in all_files:
                    all_files.append(file)
    return all_files


def get_package_model_zoo():
    cur_dir = osp.dirname(osp.realpath(__file__))
    cfg_dir = osp.join(cur_dir, "configs")
    cfgs = glob.glob(osp.join(cfg_dir, '*/*.yml'))

    valid_cfgs = []
    for cfg in cfgs:
        # exclude dataset base config
        if osp.split(osp.split(cfg)[0])[1] not in ['unittest']:
            valid_cfgs.append(cfg)
    model_names = [
        osp.relpath(cfg, cfg_dir).replace(".yml", "") for cfg in valid_cfgs
    ]

    model_zoo_file = osp.join(cur_dir, 'ppcv', 'model_zoo', 'MODEL_ZOO')
    with open(model_zoo_file, 'w') as wf:
        for model_name in model_names:
            wf.write("{}\n".format(model_name))

    return [model_zoo_file]


setup(
    name='paddlecv',
    packages=['paddlecv'],
    package_dir={'paddlecv': ''},
    package_data={
        'configs': get_package_data_files('configs', ['unittest', ]),
        'ppcv.model_zoo': get_package_model_zoo(),
    },
    include_package_data=True,
    version=VERSION,
    install_requires=requirements,
    license='Apache License 2.0',
    description='A tool for building model pipeline powered by PaddlePaddle.',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/PaddlePaddle/models',
    download_url='https://github.com/PaddlePaddle/models.git',
    keywords=['paddle-model-pipeline', 'PP-OCR', 'PP-ShiTu', 'PP-Human'],
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Utilities',
    ], )
