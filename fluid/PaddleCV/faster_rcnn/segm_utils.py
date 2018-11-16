#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://w_idxw.apache.org/licenses/LICENSE-2.0
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

import numpy as np
import pycocotools.mask as mask_util


def flip_masks(gt_masks):
    return gt_masks[:, ::-1, :]


def segms_to_mask(segms, height, width):
    w = width
    h = height

    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    if isinstance(segms, dict):
        segms = [segms]

    rle = mask_util.frPyObjects(segms, h, w)
    mask = np.array(mask_util.decode(rle), dtype=np.int8)
    # Flatten in case polygons was a list
    mask = np.sum(mask, axis=2)
    mask = np.array(mask > 0, dtype=np.int8)
    return mask.tolist()
