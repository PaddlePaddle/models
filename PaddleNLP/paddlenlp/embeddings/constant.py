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

URL_ROOT = "https://paddlenlp.bj.bcebos.com"
EMBEDDING_URL_ROOT = osp.join(URL_ROOT, "models/embeddings")

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'

EMBEDDING_NAME_LIST = [
    # baidu_encyclopedia
    "w2v.baidu_encyclopedia.target.word-word.dim300",
    "w2v.baidu_encyclopedia.target.word-character.char1-1.dim300",
    "w2v.baidu_encyclopedia.target.word-character.char1-2.dim300",
    "w2v.baidu_encyclopedia.target.word-character.char1-4.dim300",
    "w2v.baidu_encyclopedia.target.word-ngram.1-2.dim300",
    "w2v.baidu_encyclopedia.target.bigram-char.dim300",
    "w2v.baidu_encyclopedia.context.word-word.dim300",
    "w2v.baidu_encyclopedia.context.word-character.char1-1.dim300",
    "w2v.baidu_encyclopedia.context.word-character.char1-2.dim300",
    "w2v.baidu_encyclopedia.context.word-character.char1-4.dim300",
    # wikipedia
    "w2v.wiki.target.bigram-char.dim300",
    "w2v.wiki.target.word-char.dim300",
    "w2v.wiki.target.word-word.dim300",
    "w2v.wiki.target.word-bigram.dim300",
    # people_daily
    "w2v.people_daily.target.bigram-char.dim300",
    "w2v.people_daily.target.word-char.dim300",
    "w2v.people_daily.target.word-word.dim300",
    "w2v.people_daily.target.word-bigram.dim300",
    # weibo
    "w2v.weibo.target.bigram-char.dim300",
    "w2v.weibo.target.word-char.dim300",
    "w2v.weibo.target.word-word.dim300",
    "w2v.weibo.target.word-bigram.dim300",
    # sogou
    "w2v.sogou.target.bigram-char.dim300",
    "w2v.sogou.target.word-char.dim300",
    "w2v.sogou.target.word-word.dim300",
    "w2v.sogou.target.word-bigram.dim300",
    # zhihu
    "w2v.zhihu.target.bigram-char.dim300",
    "w2v.zhihu.target.word-char.dim300",
    "w2v.zhihu.target.word-word.dim300",
    "w2v.zhihu.target.word-bigram.dim300",
    # finacial
    "w2v.financial.target.bigram-char.dim300",
    "w2v.financial.target.word-char.dim300",
    "w2v.financial.target.word-word.dim300",
    "w2v.financial.target.word-bigram.dim300",
    # literature
    "w2v.literature.target.bigram-char.dim300",
    "w2v.literature.target.word-char.dim300",
    "w2v.literature.target.word-word.dim300",
    "w2v.literature.target.word-bigram.dim300",
    # siku
    "w2v.sikuquanshu.target.word-word.dim300",
    "w2v.sikuquanshu.target.word-bigram.dim300"
]
