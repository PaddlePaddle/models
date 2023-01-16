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
from ppcv.utils.download import get_dict_path

from ppcv.utils.logger import setup_logger

logger = setup_logger('Classificaion')


class Topk(object):
    def __init__(self, topk=1, class_id_map_file=None):
        assert isinstance(topk, (int, ))
        class_id_map_file = get_dict_path(class_id_map_file)
        self.class_id_map = self.parse_class_id_map(class_id_map_file)
        self.topk = topk

    def parse_class_id_map(self, class_id_map_file):
        if class_id_map_file is None:
            return None
        file_path = get_dict_path(class_id_map_file)

        try:
            class_id_map = {}
            with open(file_path, "r") as fin:
                lines = fin.readlines()
                for line in lines:
                    partition = line.split("\n")[0].partition(" ")
                    class_id_map[int(partition[0])] = str(partition[-1])
        except Exception as ex:
            msg = f"Error encountered while loading the class_id_map_file. The related setting has been ignored. The detailed error info: {ex}"
            logger.warning(msg)
            class_id_map = None
        return class_id_map

    def __call__(self, x, output_keys):
        y = []
        for idx, probs in enumerate(x):
            index = probs.argsort(axis=0)[-self.topk:][::-1].astype("int32")
            clas_id_list = []
            score_list = []
            label_name_list = []
            for i in index:
                clas_id_list.append(i.item())
                score_list.append(probs[i].item())
                if self.class_id_map is not None:
                    label_name_list.append(self.class_id_map[i.item()])
            result = {
                output_keys[0]: clas_id_list,
                output_keys[1]: np.around(
                    score_list, decimals=5).tolist(),
            }
            if label_name_list is not None:
                result[output_keys[2]] = label_name_list
            y.append(result)
        return y


class VehicleAttribute(object):
    def __init__(self, color_threshold=0.5, type_threshold=0.5):
        self.color_threshold = color_threshold
        self.type_threshold = type_threshold
        self.color_list = [
            "yellow", "orange", "green", "gray", "red", "blue", "white",
            "golden", "brown", "black"
        ]
        self.type_list = [
            "sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus",
            "truck", "estate"
        ]

    def __call__(self, x, output_keys):
        # postprocess output of predictor
        batch_res = []
        for idx, res in enumerate(x):
            res = res.tolist()
            label_res = []
            color_idx = np.argmax(res[:10])
            type_idx = np.argmax(res[10:])
            print(color_idx, type_idx)
            if res[color_idx] >= self.color_threshold:
                color_info = f"Color: ({self.color_list[color_idx]}, prob: {res[color_idx]})"
            else:
                color_info = "Color unknown"

            if res[type_idx + 10] >= self.type_threshold:
                type_info = f"Type: ({self.type_list[type_idx]}, prob: {res[type_idx + 10]})"
            else:
                type_info = "Type unknown"

            label_res = f"{color_info}, {type_info}"

            threshold_list = [self.color_threshold
                              ] * 10 + [self.type_threshold] * 9
            pred_res = (np.array(res) > np.array(threshold_list)
                        ).astype(np.int8).tolist()
            scores = np.array(res)[(
                np.array(res) > np.array(threshold_list))].tolist()
            batch_res.append({
                output_keys[0]: pred_res,
                output_keys[1]: scores,
                output_keys[2]: label_res
            })
        return batch_res


class PersonAttribute(object):
    def __init__(self,
                 threshold=0.5,
                 glasses_threshold=0.3,
                 hold_threshold=0.6):
        self.threshold = threshold
        self.glasses_threshold = glasses_threshold
        self.hold_threshold = hold_threshold

    def __call__(self, x, output_keys):
        # postprocess output of predictor
        age_list = ['AgeLess18', 'Age18-60', 'AgeOver60']
        direct_list = ['Front', 'Side', 'Back']
        bag_list = ['HandBag', 'ShoulderBag', 'Backpack']
        upper_list = ['UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice']
        lower_list = [
            'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts',
            'Skirt&Dress'
        ]
        batch_res = []
        for idx, res in enumerate(x):
            res = res.tolist()
            label_res = []
            # gender
            gender = 'Female' if res[22] > self.threshold else 'Male'
            label_res.append(gender)
            # age
            age = age_list[np.argmax(res[19:22])]
            label_res.append(age)
            # direction
            direction = direct_list[np.argmax(res[23:])]
            label_res.append(direction)
            # glasses
            glasses = 'Glasses: '
            if res[1] > self.glasses_threshold:
                glasses += 'True'
            else:
                glasses += 'False'
            label_res.append(glasses)
            # hat
            hat = 'Hat: '
            if res[0] > self.threshold:
                hat += 'True'
            else:
                hat += 'False'
            label_res.append(hat)
            # hold obj
            hold_obj = 'HoldObjectsInFront: '
            if res[18] > self.hold_threshold:
                hold_obj += 'True'
            else:
                hold_obj += 'False'
            label_res.append(hold_obj)
            # bag
            bag = bag_list[np.argmax(res[15:18])]
            bag_score = res[15 + np.argmax(res[15:18])]
            bag_label = bag if bag_score > self.threshold else 'No bag'
            label_res.append(bag_label)
            # upper
            upper_res = res[4:8]
            upper_label = 'Upper:'
            sleeve = 'LongSleeve' if res[3] > res[2] else 'ShortSleeve'
            upper_label += ' {}'.format(sleeve)
            for i, r in enumerate(upper_res):
                if r > self.threshold:
                    upper_label += ' {}'.format(upper_list[i])
            label_res.append(upper_label)
            # lower
            lower_res = res[8:14]
            lower_label = 'Lower: '
            has_lower = False
            for i, l in enumerate(lower_res):
                if l > self.threshold:
                    lower_label += ' {}'.format(lower_list[i])
                    has_lower = True
            if not has_lower:
                lower_label += ' {}'.format(lower_list[np.argmax(lower_res)])

            label_res.append(lower_label)
            # shoe
            shoe = 'Boots' if res[14] > self.threshold else 'No boots'
            label_res.append(shoe)

            threshold_list = [0.5] * len(res)
            threshold_list[1] = self.glasses_threshold
            threshold_list[18] = self.hold_threshold
            pred_res = (np.array(res) > np.array(threshold_list)
                        ).astype(np.int8).tolist()
            scores = np.array(res)[(
                np.array(res) > np.array(threshold_list))].tolist()
            batch_res.append({
                output_keys[0]: pred_res,
                output_keys[1]: scores,
                output_keys[2]: label_res,
            })
        return batch_res


class VehicleAttribute(object):
    def __init__(self, color_threshold=0.5, type_threshold=0.5):
        self.color_threshold = color_threshold
        self.type_threshold = type_threshold
        self.color_list = [
            "yellow", "orange", "green", "gray", "red", "blue", "white",
            "golden", "brown", "black"
        ]
        self.type_list = [
            "sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus",
            "truck", "estate"
        ]

    def __call__(self, x, output_keys):
        # postprocess output of predictor
        batch_res = []
        for idx, res in enumerate(x):
            res = res.tolist()
            label_res = []
            color_idx = np.argmax(res[:10])
            type_idx = np.argmax(res[10:])
            print(color_idx, type_idx)
            if res[color_idx] >= self.color_threshold:
                color_info = f"Color: ({self.color_list[color_idx]}, prob: {res[color_idx]})"
            else:
                color_info = "Color unknown"

            if res[type_idx + 10] >= self.type_threshold:
                type_info = f"Type: ({self.type_list[type_idx]}, prob: {res[type_idx + 10]})"
            else:
                type_info = "Type unknown"

            label_res = f"{color_info}, {type_info}"

            threshold_list = [self.color_threshold
                              ] * 10 + [self.type_threshold] * 9
            pred_res = (np.array(res) > np.array(threshold_list)
                        ).astype(np.int8).tolist()
            scores = np.array(res)[(
                np.array(res) > np.array(threshold_list))].tolist()
            batch_res.append({
                output_keys[0]: pred_res,
                output_keys[1]: scores,
                output_keys[2]: label_res
            })
        return batch_res


class FaceAttribute(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.label_list = np.array([
            '短胡子', '弯眉毛', '有吸引力', '眼袋', '秃顶', '刘海', '厚嘴唇', '大鼻子', '黑色头发',
            '金色头发', '模糊', '棕色头发', '浓眉毛', '胖的', '双下巴', '眼镜', '山羊胡子', '灰白头发',
            '浓妆', '高颧骨', '男性', '嘴巴微张', '胡子，髭', '小眼睛', '没有胡子', '鸭蛋脸', '皮肤苍白',
            '尖鼻子', '发际线后移', '连鬓胡子', '红润双颊', '微笑', '直发', '卷发', '戴耳环', '戴帽子',
            '涂唇膏', '戴项链', '戴领带', '年轻'
        ])

    def __call__(self, x, output_keys):
        # postprocess output of predictor
        batch_res = []
        for idx, res in enumerate(x):
            pred_idx = res > self.threshold
            pred_score = res[pred_idx]
            pred_res = self.label_list[pred_idx]

            batch_res.append({
                output_keys[0]: pred_idx.astype(np.int8).tolist(),
                output_keys[1]: pred_score.tolist(),
                output_keys[2]: pred_res.tolist()
            })
        return batch_res
