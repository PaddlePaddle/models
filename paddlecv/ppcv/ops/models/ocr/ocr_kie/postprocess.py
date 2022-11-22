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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re

from ppcv.utils.download import get_dict_path

import numpy as np
import paddle

from .preprocess import load_vqa_bio_label_maps


class VQASerTokenLayoutLMPostProcess(object):
    """ Convert between text-label and text-index """

    def __init__(self, class_path, **kwargs):
        super(VQASerTokenLayoutLMPostProcess, self).__init__()
        class_path = get_dict_path(class_path)
        label2id_map, self.id2label_map = load_vqa_bio_label_maps(class_path)

        self.label2id_map_for_draw = dict()
        for key in label2id_map:
            if key.startswith("I-"):
                self.label2id_map_for_draw[key] = label2id_map["B" + key[1:]]
            else:
                self.label2id_map_for_draw[key] = label2id_map[key]

        self.id2label_map_for_show = dict()
        for key in self.label2id_map_for_draw:
            val = self.label2id_map_for_draw[key]
            if key == "O":
                self.id2label_map_for_show[val] = key
            if key.startswith("B-") or key.startswith("I-"):
                self.id2label_map_for_show[val] = key[2:]
            else:
                self.id2label_map_for_show[val] = key

    def __call__(self, preds, batch=None, *args, **kwargs):
        if isinstance(preds, tuple):
            preds = preds[0]
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()

        if batch is not None:
            return self._metric(preds, batch[5])
        else:
            return self._infer(preds, **kwargs)

    def _metric(self, preds, label):
        pred_idxs = preds.argmax(axis=2)
        decode_out_list = [[] for _ in range(pred_idxs.shape[0])]
        label_decode_out_list = [[] for _ in range(pred_idxs.shape[0])]

        for i in range(pred_idxs.shape[0]):
            for j in range(pred_idxs.shape[1]):
                if label[i, j] != -100:
                    label_decode_out_list[i].append(self.id2label_map[label[
                        i, j]])
                    decode_out_list[i].append(self.id2label_map[pred_idxs[i,
                                                                          j]])
        return decode_out_list, label_decode_out_list

    def _infer(self, preds, segment_offset_ids, ocr_infos):
        results = []

        for pred, segment_offset_id, ocr_info in zip(preds, segment_offset_ids,
                                                     ocr_infos):
            pred = np.argmax(pred, axis=1)
            pred = [self.id2label_map[idx] for idx in pred]

            for idx in range(len(segment_offset_id)):
                if idx == 0:
                    start_id = 0
                else:
                    start_id = segment_offset_id[idx - 1]

                end_id = segment_offset_id[idx]

                curr_pred = pred[start_id:end_id]
                curr_pred = [self.label2id_map_for_draw[p] for p in curr_pred]

                if len(curr_pred) <= 0:
                    pred_id = 0
                else:
                    counts = np.bincount(curr_pred)
                    pred_id = np.argmax(counts)
                ocr_info[idx]["pred_id"] = int(pred_id)
                ocr_info[idx]["pred"] = self.id2label_map_for_show[int(
                    pred_id)]
            results.append(ocr_info)
        return results


class VQAReTokenLayoutLMPostProcess(object):
    """ Convert between text-label and text-index """

    def __init__(self, **kwargs):
        super(VQAReTokenLayoutLMPostProcess, self).__init__()

    def __call__(self, preds, label=None, *args, **kwargs):
        pred_relations = preds['pred_relations']
        if isinstance(preds['pred_relations'], paddle.Tensor):
            pred_relations = pred_relations.numpy()
        pred_relations = self.decode_pred(pred_relations)

        if label is not None:
            return self._metric(pred_relations, label)
        else:
            return self._infer(pred_relations, *args, **kwargs)

    def _metric(self, pred_relations, label):
        return pred_relations, label[-1], label[-2]

    def _infer(self, pred_relations, *args, **kwargs):
        ser_results = kwargs['ser_results']
        entity_idx_dict_batch = kwargs['entity_idx_dict_batch']

        # merge relations and ocr info
        results = []
        for pred_relation, ser_result, entity_idx_dict in zip(
                pred_relations, ser_results, entity_idx_dict_batch):
            result = []
            used_tail_id = []
            for relation in pred_relation:
                if relation['tail_id'] in used_tail_id:
                    continue
                used_tail_id.append(relation['tail_id'])
                head_idx = entity_idx_dict[relation['head_id']]
                ocr_info_head = {
                    'dt_polys': ser_result['ser.dt_polys'][head_idx].tolist(),
                    'rec_text': ser_result['ser.rec_text'][head_idx],
                    'pred': ser_result['ser.pred'][head_idx],
                }

                tail_idx = entity_idx_dict[relation['tail_id']]
                ocr_info_tail = {
                    'dt_polys': ser_result['ser.dt_polys'][tail_idx].tolist(),
                    'rec_text': ser_result['ser.rec_text'][tail_idx],
                    'pred': ser_result['ser.pred'][tail_idx],
                }
                result.append((ocr_info_head, ocr_info_tail))
            results.append(result)
        return results

    def decode_pred(self, pred_relations):
        pred_relations_new = []
        for pred_relation in pred_relations:
            pred_relation_new = []
            pred_relation = pred_relation[1:pred_relation[0, 0, 0] + 1]
            for relation in pred_relation:
                relation_new = dict()
                relation_new['head_id'] = relation[0, 0]
                relation_new['head'] = tuple(relation[1])
                relation_new['head_type'] = relation[2, 0]
                relation_new['tail_id'] = relation[3, 0]
                relation_new['tail'] = tuple(relation[4])
                relation_new['tail_type'] = relation[5, 0]
                relation_new['type'] = relation[6, 0]
                pred_relation_new.append(relation_new)
            pred_relations_new.append(pred_relation_new)
        return pred_relations_new
