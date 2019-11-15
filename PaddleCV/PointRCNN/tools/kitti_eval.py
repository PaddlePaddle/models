#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os 
from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate 

def kitti_eval():
    label_dir = os.path.join('data/KITTI/object/training', 'label_2')
    split_file = os.path.join('data/KITTI', 'ImageSets', 'val.txt')
    final_output_dir = os.path.join("./result_dir", 'final_result', 'data')
    name_to_class = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
    ap_result_str, ap_dict = kitti_evaluate(
        label_dir, final_output_dir, label_split_file=split_file,
         current_class=name_to_class["Car"])
    print("KITTI evaluate: ", ap_result_str, ap_dict)


if __name__ == "__main__":
    kitti_eval()


