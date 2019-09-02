# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import numpy as np
import logging
logger = logging.getLogger(__name__)

__all__ = [
    'get_ignore_params',
    'ignore_param_dict',
]

ignore_param_dict = {
    'FasterRCNN': ['cls_score', 'bbox_pred'],
    'CascadeRCNN': ['cls_score', 'bbox_pred'],
    'MaskRCNN': ['cls_score', 'bbox_pred', 'mask_fcn_logits'],
    'CascadeMaskRCNN': ['cls_score', 'bbox_pred', 'mask_fcn_logits'],
    'RetinaNet': ['retnet_cls_pred_fpn'],
    'SSD': ['conv2d_'],
    'YOLOv3': ['yolo_output'],
}


def get_ignore_params(arch):
    """
    In finetuning, users usually use own dataset which sets different 
    num_classes compared to published models. That will cause dimensional 
    inconsistency of parameters related to num_classes. To solve the case, 
    PaddleDetection filters the ignore params automatically. If the parameter
    name contains fields in ignore_param_dict, the parameter will not be 
    loaded.
    """
    if arch in ignore_param_dict.keys():
        return ignore_param_dict[arch]
    raise AttributeError("There is no architecture '{}'".format(arch))
