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

import numpy as np
import requests
import json
import cv2
import base64
import os


def cv2_to_base64(image):
    """cv2_to_base64

    Convert an numpy array to a base64 object.

    Args:
        image: Input array.

    Returns: Base64 output of the input.
    """
    return base64.b64encode(image).decode('utf8')


if __name__ == "__main__":
    url = "http://127.0.0.1:18093/imagenet/prediction"
    logid = 10000
    img_path = "../../images/demo.jpg"
    with open(img_path, 'rb') as file:
        image_data = file.read()
    # data should be transformed to the base64 format
    image = cv2_to_base64(image_data)
    data = {"key": ["image"], "value": [image], "logid": logid}
    # send requests
    r = requests.post(url=url, data=json.dumps(data))
    print(r.json())
