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

from . import classification
from . import detection
from . import keypoint
from . import ocr
from . import segmentation

from .classification import *
from .feature_extraction import *
from .detection import *
from .keypoint import *
from .segmentation import *
from .ocr import *

__all__ = classification.__all__ + detection.__all__ + keypoint.__all__
__all__ += segmentation.__all__
__all__ += ocr.__all__
