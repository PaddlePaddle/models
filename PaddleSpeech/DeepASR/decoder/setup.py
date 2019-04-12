#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

try:
    kaldi_root = os.environ['KALDI_ROOT']
except:
    raise ValueError("Enviroment variable 'KALDI_ROOT' is not defined. Please "
                     "install kaldi and export KALDI_ROOT=<kaldi's root dir> .")

args = [
    '-std=c++11', '-fopenmp', '-Wno-sign-compare', '-Wno-unused-variable',
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
    'tools/openfst/lib', 'src/base', 'src/matrix', 'src/util', 'src/tree',
    'src/hmm', 'src/fstext', 'src/decoder', 'src/lat'
]
LIB_DIRS = [os.path.join(kaldi_root, path) for path in LIB_DIRS]
LIB_DIRS = [os.path.abspath(path) for path in LIB_DIRS]

ext_modules = [
    Extension(
        'post_latgen_faster_mapped',
        ['pybind.cc', 'post_latgen_faster_mapped.cc'],
        include_dirs=[
            'pybind11/include', '.', os.path.join(kaldi_root, 'src'),
            os.path.join(kaldi_root, 'tools/openfst/src/include'), 'ThreadPool'
        ],
        language='c++',
        libraries=LIBS,
        library_dirs=LIB_DIRS,
        runtime_library_dirs=LIB_DIRS,
        extra_compile_args=args, ),
]

setup(
    name='post_latgen_faster_mapped',
    version='0.1.0',
    author='Paddle',
    author_email='',
    description='Decoder for Deep ASR model',
    ext_modules=ext_modules, )
