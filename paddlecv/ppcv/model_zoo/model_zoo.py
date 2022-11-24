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

import os.path as osp
import pkg_resources

try:
    from collections.abc import Sequence
except:
    from collections import Sequence

from ppcv.utils.download import get_config_path, get_model_path
from ppcv.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = [
    'list_model', 'get_config_file', 'get_model_file', 'MODEL_ZOO_FILENAME'
]

MODEL_ZOO_FILENAME = 'MODEL_ZOO'
TASK_DICT = {
    # single model
    'PP-LCNet': 'paddlecv://configs/single_op/PP-LCNet.yml',
    'PP-LCNetV2': 'paddlecv://configs/single_op/PP-LCNetV2.yml',
    'PP-HGNet': 'paddlecv://configs/single_op/PP-HGNet.yml',
    'PP-YOLOE+': 'paddlecv://configs/single_op/PP-YOLOE+.yml',
    'PP-YOLOE': 'paddlecv://configs/single_op/PP-YOLOE.yml',
    'PP-YOLO': 'paddlecv://configs/single_op/PP-YOLO.yml',
    'PP-YOLOv2': 'paddlecv://configs/single_op/PP-YOLOv2.yml',
    'PP-PicoDet': 'paddlecv://configs/single_op/PP-PicoDet.yml',
    'PP-HumanSegV2': 'paddlecv://configs/single_op/PP-HumanSegV2.yml',
    'PP-LiteSeg': 'paddlecv://configs/single_op/PP-LiteSeg.yml',
    'PP-MattingV1': 'paddlecv://configs/single_op/PP-MattingV1.yml',
    # system
    'PP-OCRv2': 'paddlecv://configs/system/PP-OCRv2.yml',
    'PP-OCRv3': 'paddlecv://configs/system/PP-OCRv3.yml',
    'PP-Structure': 'paddlecv://configs/system/PP-Structure.yml',
    'PP-ShiTuV2': 'paddlecv://configs/system/PP-ShiTuV2.yml',
    'PP-ShiTu': 'paddlecv://configs/system/PP-ShiTu.yml',
    'PP-Human': 'paddlecv://configs/system/PP-Human.yml',
    'PP-Vehicle': 'paddlecv://configs/system/PP-Vehicle.yml',
    'PP-TinyPose': 'paddlecv://configs/system/PP-TinyPose.yml',
}


def list_model(filters=[]):
    model_zoo_file = pkg_resources.resource_filename('ppcv.model_zoo',
                                                     MODEL_ZOO_FILENAME)
    with open(model_zoo_file) as f:
        model_names = f.read().splitlines()

    # filter model_name
    def filt(name):
        for f in filters:
            if name.find(f) < 0:
                return False
        return True

    if isinstance(filters, str) or not isinstance(filters, Sequence):
        filters = [filters]
    model_names = [name for name in model_names if filt(name)]
    if len(model_names) == 0 and len(filters) > 0:
        raise ValueError("no model found, please check filters seeting, "
                         "filters can be set as following kinds:\n"
                         "\tTask: single_op, system\n"
                         "\tArchitecture: PPLCNet, PPYOLOE ...\n")

    model_str = "Available Models:\n"
    for model_name in model_names:
        model_str += "\t{}\n".format(model_name)
    logger.info(model_str)


# models and configs save on bcebos under dygraph directory
def get_config_file(task):
    """Get config path from task.
    """
    if task not in TASK_DICT:
        tasks = TASK_DICT.keys()
        logger.error("Illegal task: {}, please use one of {}".format(task,
                                                                     tasks))
    path = TASK_DICT[task]
    return get_config_path(path)


def get_model_file(path):
    return get_model_path(path)
