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
"""
Utils
"""

import six
import os


def print_arguments(args):
    """
    Print arguments
    """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def mkdir(path):
    """
    Mkdir
    """
    if not os.path.isdir(path):
        if os.path.split(path)[0]:
            mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)


def pos_encoding_init():
    """
    Pos encoding init
    """
    pass


def scaled_dot_product_attention():
    """
    Scaleed dot product attention
    """
    pass
