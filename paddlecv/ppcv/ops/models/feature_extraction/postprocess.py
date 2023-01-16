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

import os

import numpy as np
import faiss
import pickle

from ppcv.utils.download import get_dict_path
from ppcv.utils.logger import setup_logger

logger = setup_logger('FeatureExtraction')


class NormalizeFeature(object):
    def __init__(self):
        super().__init__()

    def __call__(self, outputs, output_keys=None):
        features = outputs[output_keys[1]]
        feas_norm = np.sqrt(np.sum(np.square(features), axis=1, keepdims=True))
        features = np.divide(features, feas_norm)
        outputs[output_keys[1]] = features
        return outputs


class Index(object):
    def __init__(self,
                 vector_path,
                 id_map_path,
                 dist_type,
                 hamming_radius=None,
                 score_thres=None):
        vector_path = get_dict_path(vector_path)
        id_map_path = get_dict_path(id_map_path)

        if dist_type == "hamming":
            self.searcher = faiss.read_index_binary(vector_path)
        else:
            self.searcher = faiss.read_index(vector_path)

        with open(id_map_path, "rb") as fd:
            self.id_map = pickle.load(fd)

        self.dist_type = dist_type
        self.hamming_radius = hamming_radius
        self.score_thres = score_thres

    def thresh_by_score(self, output_dict, scores):
        threshed_outputs = {}
        for key in output_dict:
            threshed_outputs[key] = []
        for idx, score in enumerate(scores):
            if (self.dist_type == "hamming" and
                    score <= self.hamming_radius) or (
                        self.dist_type != "hamming" and
                        score >= self.score_thres):
                for key in output_dict:
                    threshed_outputs[key].append(output_dict[key][idx])

        return threshed_outputs

    def __call__(self, outputs, output_keys):
        features = outputs[output_keys[1]]
        scores, doc_ids = self.searcher.search(features, 1)
        docs = [self.id_map[id[0]].split()[1] for id in doc_ids]
        outputs[output_keys[2]] = [score[0] for score in scores]
        outputs[output_keys[3]] = docs

        return self.thresh_by_score(outputs, scores)


class NMS4Rec(object):
    def __init__(self, thresh):
        super().__init__()
        self.thresh = thresh

    def __call__(self, outputs, output_keys):
        bbox_list = outputs[output_keys[0]]
        x1 = np.array([bbox[0] for bbox in bbox_list])
        y1 = np.array([bbox[1] for bbox in bbox_list])
        x2 = np.array([bbox[2] for bbox in bbox_list])
        y2 = np.array([bbox[3] for bbox in bbox_list])
        scores = np.array(outputs[output_keys[2]])

        filtered_outputs = {key: [] for key in output_keys}

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        while order.size > 0:
            i = order[0]
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.thresh)[0]
            order = order[inds + 1]

            for key in output_keys:
                filtered_outputs[key].append(outputs[key][i])

        return filtered_outputs
