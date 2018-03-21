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
import glob
from distutils.core import setup, Extension
from distutils.sysconfig import get_config_vars

args = [
    '-std=c++11', '-Wno-sign-compare', '-Wno-unused-variable',
    '-Wno-unused-local-typedefs', '-Wno-unused-but-set-variable',
    '-Wno-deprecated-declarations', '-Wno-unused-function'
]

# remove warning about -Wstrict-prototypes
(opt, ) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(flag for flag in opt.split()
                             if flag != '-Wstrict-prototypes')
os.environ['CC'] = 'g++'

LIBS = [
    'fst', 'kaldi-base', 'kaldi-util', 'kaldi-matrix', 'kaldi-tree',
    'kaldi-hmm', 'kaldi-fstext', 'kaldi-decoder', 'kaldi-lat'
]

LIB_DIRS = [
    'kaldi/tools/openfst/lib', 'kaldi/src/base', 'kaldi/src/matrix',
    'kaldi/src/util', 'kaldi/src/tree', 'kaldi/src/hmm', 'kaldi/src/fstext',
    'kaldi/src/decoder', 'kaldi/src/lat'
]

ext_modules = [
    Extension(
        'post_decode_faster',
        ['pybind.cc', 'post_decode_faster.cc'],
        include_dirs=[
            'pybind11/include', '.', 'kaldi/src/',
            'kaldi/tools/openfst/src/include'
        ],
        libraries=LIBS,
        language='c++',
        library_dirs=LIB_DIRS,
        runtime_library_dirs=LIB_DIRS,
        extra_compile_args=args, ),
]

setup(
    name='post_decode_faster',
    version='0.0.1',
    author='Paddle',
    author_email='',
    description='Decoder for Deep ASR model',
    ext_modules=ext_modules, )
