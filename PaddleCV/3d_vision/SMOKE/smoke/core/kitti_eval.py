# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import csv
import logging
import subprocess
import shutil

from smoke.utils.miscellaneous import mkdir

def kitti_evaluation(dataset, predictions, output_dir):
    """Do evaluation by process kitti eval program

    Args:
        dataset (paddle.io.Dataset): [description]
        predictions (Paddle.Tensor): [description]
        output_dir (str): path of save prediction
    """
    # Clear data dir before do evaluate
    if os.path.exists(os.path.join(output_dir, 'data')):
        shutil.rmtree(os.path.join(output_dir, 'data'))
    predict_folder = os.path.join(output_dir, 'data')  # only recognize data
    mkdir(predict_folder)
    type_id_conversion = getattr(dataset, 'TYPE_ID_CONVERSION')
    id_type_conversion = {value:key for key, value in type_id_conversion.items()}
    for image_id, prediction in predictions.items():
        predict_txt = image_id + '.txt'
        predict_txt = os.path.join(predict_folder, predict_txt)

        generate_kitti_3d_detection(prediction, predict_txt, id_type_conversion)
    
    output_dir = os.path.abspath(output_dir)
    root_dir = os.getcwd()
    os.chdir('./tools/kitti_eval_offline')
    label_dir = getattr(dataset, 'label_dir')
    label_dir = os.path.join(root_dir, label_dir)

    if not os.path.isfile('evaluate_object_3d_offline'):
        subprocess.Popen('g++ -O3 -DNDEBUG -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp', shell=True)
    command = "./evaluate_object_3d_offline {} {}".format(label_dir, output_dir)

    os.system(command)

def generate_kitti_3d_detection(prediction, predict_txt, id_type_conversion):
    """write kitti 3d detection result to txt file 

    Args:
        prediction (list[float]): final prediction result
        predict_txt (str): path to save the result
    """
    with open(predict_txt, 'w', newline='') as f:
        w = csv.writer(f, delimiter=' ', lineterminator='\n')
        if len(prediction) == 0:
            w.writerow([])
        else:
            for p in prediction:
                p = p.round(4)
                type = id_type_conversion[int(p[0])]
                row = [type, 0, 0] + p[1:].tolist()
                w.writerow(row)

    check_last_line_break(predict_txt)

def check_last_line_break(predict_txt):
    """check predict last lint

    Args:
        predict_txt (str): path of predict txt
    """
    f = open(predict_txt, 'rb+')
    try:
        f.seek(-1, os.SEEK_END)
    except:
        pass
    else:
        if f.__next__() == b'\n':
            f.seek(-1, os.SEEK_END)
            f.truncate()
    f.close()