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


class InformationExtractionDecode(object):
    def __init__(self):
        pass

    def __call__(self, preds, output_keys):
        results = []
        for batch_idx, pred in enumerate(preds):
            type_list = []
            txt_list = []
            for k, v_list in pred.items():
                type_list.append(k)
                txt_list.append([v['text'] for v in v_list])
            results.append({
                output_keys[0]: txt_list,
                output_keys[1]: type_list
            })
        return results


class SentimentAnalysisDecode(object):
    def __init__(self):
        pass

    def __call__(self, preds, output_keys):
        results = []
        for batch_idx, pred in enumerate(preds):
            results.append({output_keys[0]: pred['label']})
        return results
