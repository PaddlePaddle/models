# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import cv2
import numpy as np
from scipy.special import softmax

from ppcv.utils.download import get_dict_path

__all__ = ['ParserTrackerResults']


class ParserTrackerResults(object):
    def __init__(self, label_list):
        self.clsid2catid, self.catid2name = self.get_categories(label_list)

    def get_categories(self, label_list):
        if isinstance(label_list, list):
            clsid2catid = {i: i for i in range(len(label_list))}
            catid2name = {i: label_list[i] for i in range(len(label_list))}
            return clsid2catid, catid2name

        label_list = get_dict_path(label_list)
        if label_list.endswith('json'):
            # lazy import pycocotools here
            from pycocotools.coco import COCO
            coco = COCO(label_list)
            cats = coco.loadCats(coco.getCatIds())
            clsid2catid = {i: cat['id'] for i, cat in enumerate(cats)}
            catid2name = {cat['id']: cat['name'] for cat in cats}
        elif label_list.endswith('txt'):
            cats = []
            with open(label_list) as f:
                for line in f.readlines():
                    cats.append(line.strip())
            if cats[0] == 'background': cats = cats[1:]

            clsid2catid = {i: i for i in range(len(cats))}
            catid2name = {i: name for i, name in enumerate(cats)}

        else:
            raise ValueError("label_list {} should be json or txt.".format(
                label_list))
        return clsid2catid, catid2name

    def __call__(self, tracking_outputs, output_keys):
        tk_cls_ids = tracking_outputs[output_keys[3]]
        tk_cls_names = [
            self.catid2name[self.clsid2catid[cls_id]] for cls_id in tk_cls_ids
        ]
        tracking_outputs[output_keys[4]] = tk_cls_names
        return tracking_outputs
