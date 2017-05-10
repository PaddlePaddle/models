# -*- encoding:utf-8 -*-
# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
import gzip
import paddle.v2 as paddle
from nce_conf import network_conf


def main():
    paddle.init(use_gpu=False, trainer_count=1)
    word_dict = paddle.dataset.imikolov.build_dict()
    dict_size = len(word_dict)

    prediction_layer = network_conf(
        is_train=False, hidden_size=256, embed_size=32, dict_size=dict_size)

    with gzip.open("model_params.tar.gz", 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    idx_word_dict = dict((v, k) for k, v in word_dict.items())
    batch_size = 64
    batch_ins = []
    ins_iter = paddle.dataset.imikolov.test(word_dict, 5)

    infer_data = []
    infer_label_data = []
    cnt = 0
    for item in paddle.dataset.imikolov.test(word_dict, 5)():
        infer_data.append((item[:4]))
        infer_label_data.append(item[4])
        cnt += 1
        if cnt == 100:
            break

    predictions = paddle.infer(
        output_layer=prediction_layer, parameters=parameters, input=infer_data)

    for i, prob in enumerate(predictions):
        print prob, infer_label_data[i]


if __name__ == '__main__':
    main()
