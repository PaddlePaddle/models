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

import sys
import numpy as np
import base64
from PIL import Image
import io
from preprocess_ops import ResizeImage, CenterCropImage, NormalizeImage, ToCHW, Compose
from paddle_serving_client import Client

url = "127.0.0.1:9997"
logid = 10000
img_path = "../../images/demo.jpg"

def preprocess():
    image_file = img_path
    image = Image.open(image_file)
    seq = Compose([
            ResizeImage(256), CenterCropImage(224), NormalizeImage(), ToCHW()
        ])
    input_data=seq(image)
    feed = {"input": input_data}
    fetch = ["softmax_1.tmp_0"]
    return feed, fetch

def postprocess(fetch_map):
    score_list = fetch_map["softmax_1.tmp_0"]
    fetch_dict = {"class_id": [], "prob": []}
    for score in score_list:
        score = score.tolist()
        max_score = max(score)
        fetch_dict["class_id"].append(score.index(max_score))
        fetch_dict["prob"].append(max_score)
    fetch_dict["class_id"] = str(fetch_dict["class_id"])
    fetch_dict["prob"] = str(fetch_dict["prob"])
    return fetch_dict

client = Client()
client.load_client_config(
    "serving_client/serving_client_conf.prototxt")
client.connect([url])

feed, fetch = preprocess()

fetch_map = client.predict(feed=feed, fetch=fetch)

result = postprocess(fetch_map)
print(result)