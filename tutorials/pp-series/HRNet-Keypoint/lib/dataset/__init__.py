# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from . import category
from . import dataset
from . import keypoint_coco
from . import reader
from . import transform

from .category import *
from .dataset import *
from .keypoint_coco import *
from .reader import *
from .transform import *

__all__ = category.__all__ + dataset.__all__ + keypoint_coco.__all__ \
          + reader.__all__  + transform.__all__
