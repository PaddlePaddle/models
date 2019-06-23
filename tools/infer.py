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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from paddle import fluid

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.modeling.model_input import create_feeds
from ppdet.data.data_feed import create_reader

from ppdet.utils.eval_utils import parse_fetches
from ppdet.utils.cli import parse_args
from ppdet.utils.visualizer import visualize_results
import ppdet.utils.checkpoint as checkpoint

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if 'architecture' in cfg:
        main_arch = cfg['architecture']
    else:
        raise ValueError("'architecture' not specified in config file.")

    merge_config(args.cli_config)

    if 'test_feed' not in cfg:
        test_feed = create(type(main_arch).__name__ + 'TestFeed')
    else:
        test_feed = create(cfg['test_feed'])

    place = fluid.CUDAPlace(0) if cfg['use_gpu'] else fluid.CPUPlace()
    exe = fluid.Executor(place)

    model = create(main_arch)

    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            _, feed_vars = create_feeds(test_feed, use_pyreader=False)
            test_fetches = model.test(feed_vars)
    infer_prog = infer_prog.clone(True)

    reader = create_reader(test_feed)
    feeder = fluid.DataFeeder(place=place, feed_list=feed_vars.values())

    exe.run(startup_prog)
    if cfg['weights']:
        checkpoint.load_checkpoint(exe, infer_prog, cfg['weights'])

    # parse infer fetches
    extra_keys = []
    if cfg['metric'] == 'COCO':
        extra_keys = ['im_info', 'im_id', 'im_shape']
    keys, values, _ = parse_fetches(test_fetches, infer_prog, extra_keys)

    # 6. Parse dataset category
    if cfg['metric'] == 'COCO':
        from ppdet.utils.coco_eval import bbox2out, mask2out, get_category_info
    if cfg['metric'] == "VOC":
        # TODO(dengkaipeng): add VOC metric process
        pass

    anno_file = getattr(test_feed.dataset, 'annotation', None)
    with_background = getattr(test_feed, 'with_background', True)
    clsid2catid, catid2name = get_category_info(anno_file, with_background)

    imid2path = reader.imid2path
    for iter_id, data in enumerate(reader()):
        outs = exe.run(infer_prog,
                       feed=feeder.feed(data),
                       fetch_list=values,
                       return_numpy=False)
        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(keys, outs)
        }
        logger.info('Infer iter {}'.format(iter_id))

        im_id = int(res['im_id'][0])
        image_path = os.path.join(test_feed.dataset.image_dir, imid2path[im_id])
        if cfg['metric'] == 'COCO':
            bbox_results = None
            mask_results = None
            if 'bbox' in res:
                bbox_results = bbox2out([res], clsid2catid)
            if 'mask' in res:
                mask_results = mask2out([res], clsid2catid,
                                        cfg['MaskHead']['resolution'])
            visualize_results(image_path, catid2name, 0.5, bbox_results,
                              mask_results)

        if cfg['metric'] == "VOC":
            # TODO(dengkaipeng): add VOC metric process
            pass


if __name__ == '__main__':
    main()
