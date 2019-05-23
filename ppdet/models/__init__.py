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

from . import backbones
from .backbones import *

from . import anchor_heads
#from .anchor_heads import *

from . import detectors
from .detectors import *

from . import roi_extractors
from .roi_extractors import *

from . import bbox_heads
from .bbox_heads import *

from . import registry
from .registry import *

__all__ = backbones.__all__
__all__ += anchor_heads.__all__
__all__ += roi_extractors.__all__
__all__ += bbox_heads.__all__
__all__ += detectors.__all__
__all__ += registry.__all__
