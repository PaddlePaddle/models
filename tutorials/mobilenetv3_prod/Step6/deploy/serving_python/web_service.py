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

from paddle_serving_server.web_service import WebService, Op

import sys
import numpy as np
import base64
from PIL import Image
import io
from preprocess_ops import ResizeImage, CenterCropImage, NormalizeImage, ToCHW, Compose

class MobileNetV3Op(Op):
    def init_op(self):
        self.seq = Compose([
            ResizeImage(256), CenterCropImage(224), NormalizeImage(), ToCHW()
        ])

    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        batch_size = len(input_dict.keys())
        imgs = []
        for key in input_dict.keys():
            data = base64.b64decode(input_dict[key].encode('utf8'))
            byte_stream = io.BytesIO(data)
            img = Image.open(byte_stream)
            img = img.convert("RGB")
            img = self.seq(img)
            imgs.append(img[np.newaxis, :].copy())
        input_imgs = np.concatenate(imgs, axis=0)
        return {"input": input_imgs}, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        score_list = fetch_dict["softmax_1.tmp_0"]
        result = {"class_id": [], "prob": []}
        for score in score_list:
            score = score.tolist()
            max_score = max(score)
            result["class_id"].append(score.index(max_score))
            result["prob"].append(max_score)
        result["class_id"] = str(result["class_id"])
        result["prob"] = str(result["prob"])
        return result, None, ""

class MobileNetV3Service(WebService):
    def get_pipeline_response(self, read_op):
        mobilenetv3_op = MobileNetV3Op(
            name="imagenet", input_ops=[read_op])
        return mobilenetv3_op


# define the service class
uci_service = MobileNetV3Service(name="imagenet")
# load config and prepare the service
uci_service.prepare_pipeline_config("config.yml")
# start the service
uci_service.run_service()
