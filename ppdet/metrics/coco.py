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
from __future__ import unicode_literals

import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import logging
logger = logging.getLogger(__name__)

__all__ = ['coco_eval']


def coco_eval(results, anno_file, outfile):
    coco_gt = COCO(anno_file)

    clsid2catid = {i + 1: v for i, v in enumerate(coco_gt.getCatIds())}

    if 'bbox' in results[0]:
        xywh_results = bbox2out(results, clsid2catid)

        assert outfile.endswith('.json')
        with open(outfile, 'w') as f:
            json.dump(xywh_results, f)

        logger.info("Start evaluate...")
        coco_dt = coco_gt.loadRes(outfile)
        coco_ev = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_ev.evaluate()
        coco_ev.accumulate()
        coco_ev.summarize()
    #if 'mask' in results[0]:
    #    #TODO: eval sgem


def bbox2out(results, clsid2catid):
    xywh_res = []
    for t in results:
        bboxes = t['bbox'][0]
        lengths = t['bbox'][1][0]
        im_ids = np.array(t['im_id'][0])
        if bboxes.shape == (1, 1) or bboxes is None:
            continue

        k = 0
        for i in range(len(lengths)):
            num = lengths[i]
            im_id = int(im_ids[i][0])
            for j in range(num):
                dt = bboxes[k]
                clsid, score, xmin, ymin, xmax, ymax = dt.tolist()
                catid = clsid2catid[clsid]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                bbox = [xmin, ymin, w, h]
                coco_res = {
                    'image_id': im_id,
                    'category_id': catid,
                    'bbox': bbox,
                    'score': score
                }
                xywh_res.append(coco_res)
                k += 1
    return xywh_res
