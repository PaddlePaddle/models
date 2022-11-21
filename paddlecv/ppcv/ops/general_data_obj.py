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

import numby as np


class GeneralDataObj(object):
    def __init__(self, data):
        assert isinstance(data, (dict, ))
        self.data_dict = data
        pass

    def get(self, key):
        """
        key can be one of [list, tuple, str]
        """
        if isinstance(key, (list, tuple)):
            return [self.data_dict[k] for k in key]
        elif isinstance(key, (str)):
            return self.data_dict[key]
        else:
            assert False, f"key({key}) type must be in on of [list, tuple, str] but got {type(key)}"

    def set(self, key, value):
        """
        key: str
        value: an object
        """
        self.data_dict[key] = value

    def keys(self, ):
        """
        get all keys of the data
        """
        return list(self.data_dict.keys())
