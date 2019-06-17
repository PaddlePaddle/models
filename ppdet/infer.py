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

import paddle.fluid as fluid
import sys
import numpy as np
import logging
from pycocotools.coco import COCO
from args import parse_args, print_arguments
from ppdet.core.config import load_cfg, merge_cfg
from ppdet.models import Detectors
import ppdet.utils.checkpoint as checkpoint
from ppdet.dataset.reader import Reader
#from visualizer import get_color, visual_single_img
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
    detector = Detectors.get(cfg.MODEL.TYPE)(cfg)
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
    keys, values = [], []
    for k, v in fetches.items():
        keys.append(k)
        values.append(v.name)
    if cfg.TEST.METRIC_TYPE == 'COCO':
        for v in ['im_info', 'im_id']:
            try:
                if fluid.framework._get_var(v, test_prog):
                    keys += [v]
                    values += [v]
            except:
                pass
    #print("the final key",keys,values)
    for v in values:
        fluid.framework._get_var(str(v), test_prog).persistable = True

    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = False
    compile_program = fluid.compiler.CompiledProgram(
        test_prog).with_data_parallel(build_strategy=build_strategy)

    # 4. Define reader
    reader = Reader(cfg.DATA, cfg.TRANSFORM)
    test_reader = reader.test()
    pyreader = detector.get_pyreader()
    pyreader.decorate_sample_list_generator(test_reader, place)

    # 5. Load model
    exe.run(startup_prog)
    if cfg.TEST.WEIGHTS:
        checkpoint.load(exe, test_prog, cfg.TEST.WEIGHTS)

    # 6. Run
    iter_id = 1
    results = []
    file_list = cfg.DATA.TEST.ANNO_FILE
    with_background = getattr(cfg.DATA.TEST, 'WITH_BACKGROUND', True)
    a = True
    try:
        pyreader.start()
        while a:
            outs = exe.run(compile_program,
                           fetch_list=values,
                           return_numpy=False)
            res = {
                k: (np.array(v), v.recursive_sequence_lengths())
                for k, v in zip(keys, outs)
            }
            results.append(res)
            # only infer first pics
            if iter_id > 3:
                a = False
            if iter_id % 10 == 0:
                logger.info('Test iter {}'.format(iter_id))
            iter_id += 1
    except (StopIteration, fluid.core.EOFException):
        pyreader.reset()
    logger.info('Test iter {}'.format(iter_id))

    # infer
    if cfg.TEST.METRIC_TYPE == 'COCO':
        coco = COCO(file_list)
        cat_ids = coco.loadCats(coco.getCatIds())
        clsid2catid = dict(
                {i + int(with_background): catid
                    for i, catid in enumerate(cat_ids)})
        from ppdet.metrics.coco import bbox2out
        xywh_results = bbox2out(results, clsid2catid)
        img = -1
        for i in range(len(xywh_results)):
            bboxes = xywh_results[i]
            labels = bboxes['category_id']['name']
            img_id = bboxes['image_id']
            scores = bboxes['score']
            boxes = bboxes['bbox']
            if scores < 0.5:
                pass
            else:
                if img != img_id: 
                    print("img_name:", img_id)
                    img = img_id
                print("\t {:15s} at {:25} score: {:.5f}".format(labels, str(list(map(int, list(boxes)))), scores))
                #color_list = get_color(category_num=80)
                #visual_single_img(img, bbox_per_img, segm_per_img, img_name, output_folder,color_list, category_name, show_border, thresh)
                # todo: add visualizer

if __name__ == '__main__':
    main()
