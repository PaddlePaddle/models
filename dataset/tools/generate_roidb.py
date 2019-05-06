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

# function:
#   tool used to convert roidb data in json to pickled file

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys
import pickle as pkl

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
if path not in sys.path:
    sys.path.insert(0, path)

from dataset.source.roidb_loader import load

def parse_args():
    """ parse arguments
    """
    parser = argparse.ArgumentParser(
        description='Generate SimpleDet GroundTruth Database')

    parser.add_argument('--annotation', help='json file name for annotation', type=str)
    parser.add_argument('--save-dir', type=str,
        help='directory to save roidb files', default='data/tests')
    parser.add_argument('--samples', default=-1, type=int,
        help='number of samples to dump,default to all')

    args = parser.parse_args()
    return args


def dump_json_as_pickle(args):
    """ tool used to load json data, and then save it as pickled file

    """
    samples = args.samples 
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    annotation_path = args.annotation

    roidb = load(annotation_path, samples)

    samples = len(roidb)
    dsname = os.path.basename(annotation_path).rstrip('.json')
    roidb_fname = save_dir + "/%s.roidb" % (dsname)
    with open(roidb_fname, "wb") as fout:
        pkl.dump(roidb, fout)

    print('dumped %d samples to file[%s]' % (samples, roidb_fname))


if __name__ == "__main__":
    """ make sure your data is stored in 'data/${args.dataset}'

    usage:
        python generate_roidb.py --annotation=./annotations/instances_val2017.json
            --save-dir=./test/data --sample=100
    """
    args = parse_args()
    dump_json_as_pickle(args)
