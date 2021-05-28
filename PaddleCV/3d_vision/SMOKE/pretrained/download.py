# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
from io import BytesIO
import urllib.request
from zipfile import ZipFile

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    file_url = "https://bj.bcebos.com/paddleseg/3d/smoke/dla34.pdparams"
    urllib.request.urlretrieve(file_url, "dla34.pdparams")

    smoke_model_path = 'https://bj.bcebos.com/paddleseg/3d/smoke/smoke-release.zip'
    with urllib.request.urlopen(smoke_model_path) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(LOCAL_PATH)