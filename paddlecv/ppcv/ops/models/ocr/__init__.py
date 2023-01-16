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

from . import ocr_db_detection
from . import ocr_crnn_recognition
from . import ocr_table_recognition
from . import ocr_kie

from .ocr_db_detection import *
from .ocr_crnn_recognition import *
from .ocr_table_recognition import *
from .ocr_kie import *

__all__ = ocr_db_detection.__all__
__all__ += ocr_crnn_recognition.__all__
__all__ += ocr_table_recognition.__all__
__all__ += ocr_kie.__all__
