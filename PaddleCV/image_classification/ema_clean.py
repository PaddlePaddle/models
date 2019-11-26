#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import functools
import shutil
from utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
# yapf: disable
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('ema_model_dir',     str, None, "The directory of model which use ExponentialMovingAverage to train")
add_arg('cleaned_model_dir', str, None, "The directory of cleaned model")
# yapf: enable

def main():
    args = parser.parse_args()
    print_arguments(args)
    if not os.path.exists(args.cleaned_model_dir):
        os.makedirs(args.cleaned_model_dir)

    items = os.listdir(args.ema_model_dir)
    for item in items:
        if item.find('ema') > -1:
            item_clean = item.replace('_ema_0', '')
            shutil.copyfile(os.path.join(args.ema_model_dir, item),
                            os.path.join(args.cleaned_model_dir, item_clean))
        elif item.find('mean') > -1 or item.find('variance') > -1:
            shutil.copyfile(os.path.join(args.ema_model_dir, item),
                            os.path.join(args.cleaned_model_dir, item))

if __name__ == '__main__':
    main()
