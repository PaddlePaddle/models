# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from enum import Enum
import os.path as osp

URL_ROOT = "https://bj.bcebos.com/paddlenlp"
EMBEDDING_URL_ROOT = osp.join(URL_ROOT, "models/embeddings")

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'

PAD_IDX = 0
UNK_IDX = 1

EMBEDDING_NAME_LIST = ["w2v.baidu_encyclopedia.target.word-word.dim300"]
