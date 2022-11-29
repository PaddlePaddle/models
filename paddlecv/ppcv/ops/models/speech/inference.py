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

import importlib
import pathlib
import os

from ppcv.ops.base import create_operators, BaseOp
from ppcv.core.workspace import register
from ppcv.utils.utility import check_install


@register
class TTSOp(BaseOp):
    def __init__(self, model_cfg, env_cfg):
        super(TTSOp, self).__init__(model_cfg, env_cfg)
        check_install('paddlespeech', 'paddlespeech')
        from paddlespeech.cli.tts import TTSExecutor

        mod = importlib.import_module(__name__)
        env_cfg["batch_size"] = model_cfg.get("batch_size", 1)
        self.batch_size = env_cfg["batch_size"]
        self.name = model_cfg["name"]
        self.frame = -1
        keys = self.get_output_keys()
        self.output_keys = [self.name + '.' + key for key in keys]

        self.tts = TTSExecutor()
        self.output_dir = self.env_cfg.get('output_dir', 'output')

    @classmethod
    def get_output_keys(cls):
        return ["fn"]

    @classmethod
    def type(self):
        return 'MODEL'

    def infer(self, inputs):
        results = []
        for data in inputs:
            img_path = data[self.input_keys[0]]
            txts = data[self.input_keys[1]]
            save_path = os.path.join(self.output_dir,
                                     pathlib.Path(img_path).stem + '.wav')
            # model inference
            self.tts(text=''.join(txts),
                     output=save_path,
                     am='fastspeech2_mix',
                     voc='hifigan_csmsc',
                     lang='mix',
                     spk_id=174)
            results.append({self.output_keys[0]: save_path})
        return results

    def __call__(self, inputs):
        """
        step1: parser inputs
        step2: run
        step3: merge results
        input: a list of dict
        """
        # step2: run
        outputs = self.infer(inputs)
        return outputs
