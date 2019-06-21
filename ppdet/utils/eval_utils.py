"""
  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# TODO(dengkaipeng): change comment stype above in github

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

import paddle.fluid as fluid

__all__ = ['parse_fetches', 'eval_run', 'eval_results']

logger = logging.getLogger(__name__)


def parse_fetches(fetches, prog=None, extra_keys=None):
    """
    Parse fetch variable infos from model fetches,
    values for fetch_list and keys for stat
    """
    keys, values = [], []
    for k, v in fetches.items():
        keys.append(k)
        v.persistable = True
        values.append(v.name)

    if prog is not None and extra_keys is not None:
        for k in extra_keys:
            try:
                v = fluid.framework._get_var(k, prog)
                v.persistable = True
                keys.append(k)
                values.append(v.name)
            except:
                pass

    return keys, values


def eval_run(exe, compile_program, pyreader, keys, values):
    """
    Run evaluation program, return program outputs.
    """
    iter_id = 0
    results = []
    try:
        pyreader.start()
        while True:
            outs = exe.run(compile_program,
                           fetch_list=values,
                           return_numpy=False)
            res = {
                k: (np.array(v), v.recursive_sequence_lengths())
                for k, v in zip(keys, outs)
            }
            results.append(res)
            if iter_id % 100 == 0:
                logger.info('Test iter {}'.format(iter_id))
            iter_id += 1
    except (StopIteration, fluid.core.EOFException):
        pyreader.reset()
    logger.info('Test finish iter {}'.format(iter_id))

    return results


def eval_results(results, feed, args, cfg):
    """
    Evaluation for evaluation program results
    """
    metric = cfg['metric']
    if metric == 'COCO':
        from ppdet.utils.coco_eval import bbox_eval, mask_eval
        anno_file = getattr(feed.dataset, 'annotation', None)
        with_background = getattr(feed, 'with_background', True)
        savefile = 'bbox.json'
        if args.savefile:
            savefile = '{}_bbox.json'.format(args.savefile)
        bbox_eval(results, anno_file, savefile, with_background)
        if 'mask' in results[0]:
            savefile= 'mask.json'
            if args.savefile:
                savefile = '{}_mask.json'.format(args.savefile)
            mask_eval(results, anno_file, savefile,
                      cfg['MaskHead']['resolution'])
    else:
        res = np.mean(results[-1]['map'][0])
        logger.info('Test mAP: {}'.format(res))

