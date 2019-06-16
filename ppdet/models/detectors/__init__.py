#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from . import faster_rcnn
from .faster_rcnn import *

from . import mask_rcnn
from .mask_rcnn import *

from . import yolov3
from .yolov3 import *

from . import ssd
from .ssd import *

from . import retinanet
from .retinanet import *

__all__ = faster_rcnn.__all__
__all__ = mask_rcnn.__all__
__all__ += yolov3.__all__
__all__ += ssd.__all__
__all__ += retinanet.__all__
