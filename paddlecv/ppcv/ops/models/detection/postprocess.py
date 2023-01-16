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


class ParserDetResults(object):
    def __init__(self,
                 label_list,
                 threshold=0.5,
                 max_det_results=100,
                 keep_cls_ids=None):
        self.threshold = threshold
        self.max_det_results = max_det_results
        self.clsid2catid, self.catid2name = self.get_categories(label_list)
        self.keep_cls_ids = keep_cls_ids if keep_cls_ids else list(
            self.clsid2catid.keys())

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

    def __call__(self, preds, bbox_num, output_keys):
        start_id = 0
        dt_bboxes = []
        scores = []
        class_ids = []
        cls_names = []
        new_bbox_num = []

        for num in bbox_num:
            end_id = start_id + num
            pred = preds[start_id:end_id]
            start_id = end_id
            max_det_results = min(self.max_det_results, pred.shape[0])
            keep_indexes = pred[:, 1].argsort()[::-1][:max_det_results]

            select_num = 0
            for idx in keep_indexes:
                single_res = pred[idx].tolist()
                class_id = int(single_res[0])
                score = single_res[1]
                bbox = single_res[2:]
                if score < self.threshold:
                    continue
                if class_id not in self.keep_cls_ids:
                    continue
                select_num += 1
                dt_bboxes.append(bbox)
                scores.append(score)
                class_ids.append(class_id)
                cls_names.append(self.catid2name[self.clsid2catid[class_id]])
            new_bbox_num.append(select_num)
        result = {
            output_keys[0]: dt_bboxes,
            output_keys[1]: scores,
            output_keys[2]: class_ids,
            output_keys[3]: cls_names,
        }
        new_bbox_num = np.array(new_bbox_num).astype('int32')
        return result, new_bbox_num