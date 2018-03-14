#  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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

import os
from distutils.core import setup, Extension
from distutils.sysconfig import get_config_vars

args = ['-std=c++11']

# remove warning about -Wstrict-prototypes
(opt, ) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(flag for flag in opt.split()
                             if flag != '-Wstrict-prototypes')

ext_modules = [
    Extension(
        'decoder',
        ['pybind.cc', 'decoder.cc'],
        include_dirs=['pybind11/include', '.'],
        language='c++',
        extra_compile_args=args, ),
]

setup(
    name='decoder',
    version='0.0.1',
    author='Paddle',
    author_email='',
    description='Decoder for Deep ASR model',
    ext_modules=ext_modules, )
