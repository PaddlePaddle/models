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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
from pycocotools.coco import COCO

import logging
logging.basicConfig(level=logging.INFO)

import paddle.fluid as fluid

from ppdet.core.config import load_cfg, merge_cfg
from ppdet.models import Detectors
import ppdet.utils.checkpoint as checkpoint
from ppdet.dataset.reader import Reader
from ppdet.utils.run_utils import parse_fetches
from ppdet.utils.visualizer import visualize_results
from args import parse_args, print_arguments

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    """ infer image from file
    """
    # 1. load config
    args = parse_args()
    print_arguments(args)
    if args.cfg_file is None:
        raise ValueError("Should specify --cfg_file=configure_file_path.")
    cfg = load_cfg(args.cfg_file)
    merge_cfg(vars(args), cfg)
    merge_cfg({'IS_TRAIN': False}, cfg)

    if cfg.TEST.METRIC_TYPE == 'VOC':
        merge_cfg({'MODE': 'val'}, cfg)

    # 2. build program
    # get detector and losses
    detector = Detectors.get(cfg.MODEL.TYPE)(cfg, use_pyreader=False)
    startup_prog = fluid.Program()
    test_prog = fluid.Program()
    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            if cfg.TEST.METRIC_TYPE == 'COCO':
                fetches = detector.test()
            else:
                fetches = detector.val()
    test_prog = test_prog.clone(True)

    # define executor
    place = fluid.CUDAPlace(0) if cfg.ENV.GPU else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # 3. Compile program for multi-devices
    extra_keys = ['im_info', 'im_id'] if cfg.TEST.METRIC_TYPE == 'COCO' \
                 else []
    keys, values = parse_fetches(fetches, test_prog, extra_keys)

    # 4. Define reader
    reader = Reader(cfg.DATA, cfg.TRANSFORM)
    test_reader = reader.test()
    feed_list = detector.get_feed_list()
    feeder = fluid.DataFeeder(place=place, feed_list=feed_list)

    # 5. Load model
    exe.run(startup_prog)
    if cfg.TEST.WEIGHTS:
        checkpoint.load(exe, test_prog, cfg.TEST.WEIGHTS)

    # 6. Parse dataset category
    if cfg.TEST.METRIC_TYPE == 'COCO':
        from ppdet.metrics.coco import bbox2out, mask2out, get_category_info
    if cfg.TEST.METRIC_TYPE == "VOC":
        # TODO(dengkaipeng): add VOC metric process
        pass

    anno_file = getattr(cfg.DATA.TEST, 'ANNO_FILE', None)
    with_background = getattr(cfg.DATA.TEST, 'WITH_BACKGROUND', True)
    clsid2catid, catid2name = get_category_info(anno_file, with_background)

    # 7. Run
    iter_id = 0
    # FIXME(dengkaipeng): cannot pass image_path through feed data,
    # get image pathes from TEST_FILE and index path with im_id, 
    # better impliment is required
    with open(cfg.DATA.TEST.TEST_FILE) as f:
        images = f.readlines()

    for iter_id, data in enumerate(test_reader()):
        outs = exe.run(test_prog,
                       feed=feeder.feed(data),
                       fetch_list=values,
                       return_numpy=False)
        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(keys, outs)
        }
        logger.info('Infer iter {}'.format(iter_id))

        im_id = int(res['im_id'][0])
        image_path = os.path.join(cfg.DATA.TEST.IMAGE_DIR, images[im_id].strip())
        if cfg.TEST.METRIC_TYPE == 'COCO':
            bbox_results = bbox2out([res], clsid2catid) \
                                if 'bbox' in res else None
            mask_results = mask2out([res], clsid2catid, cfg.MASK_HEAD.RESOLUTION) \
                                if 'mask' in res else None
            visualize_results(image_path, catid2name, 0.5, bbox_results, mask_results)
        if cfg.TEST.METRIC_TYPE == "VOC":
            # TODO(dengkaipeng): add VOC metric process
            pass


if __name__ == '__main__':
    main()
