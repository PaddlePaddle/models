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

from paddle_serving_client import Client

url = "127.0.0.1:9993"
logid = 10000
img_path = "./images/demo.jpg"

def preprocess():
    """preprocess
        
    In preprocess stage, assembling data for process stage. users can 
    override this function for model feed features.

    Return:
        feed: input data for inference
        fetch: name list of output
    """
    pass

def postprocess(fetch_map):
    """postprocess

    In postprocess stage, assemble data for output.
    Args:
        fetch_map: data returned in process stage, dict(for single predict)

    Returns: 
        fetch_dict: fetch result must be dict type.
    """
    pass

client = Client()
client.load_client_config(
    "serving_client/serving_client_conf.prototxt")
client.connect([url])

feed, fetch = preprocess()

fetch_map = client.predict(feed=feed, fetch=fetch)

result = postprocess(fetch_map)
print(result)