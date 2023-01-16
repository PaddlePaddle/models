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

from .base import OutputBaseOp
from .classification import ClasOutput
from .feature_extraction import FeatureOutput
from .detection import DetOutput
from .keypoint import KptOutput
from .ocr import OCRTableOutput, OCROutput, PPStructureOutput, PPStructureReOutput, PPStructureSerOutput
from .segmentation import SegOutput, HumanSegOutput, MattingOutput
from .tracker import TrackerOutput

__all__ = [
    'OutputBaseOp', 'ClasOutput', 'FeatureOutput', 'DetOutput', 'KptOutput',
    'SegOutput', 'HumanSegOutput', 'MattingOutput', 'OCROutput',
    'OCRTableOutput', 'PPStructureOutput', 'PPStructureReOutput',
    'PPStructureSerOutput', 'TrackerOutput'
]
