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

import pickle

from paddle.utils import download
from paddleaudio.utils.log import logger
from utils import get_pickle_results, get_txt_from_url

URL = {
    'zs_feature':
    'https://bj.bcebos.com/paddleaudio/examples/dcase21_task1b/clip_visual_zs.pkl',
    'train_split':
    'https://bj.bcebos.com/paddleaudio/examples/dcase21_task1b/train_split.txt',
    'eval_split':
    'https://bj.bcebos.com/paddleaudio/examples/dcase21_task1b/eval_split.txt'
}

if __name__ == '__main__':
    results = get_pickle_results(URL['zs_feature'])
    eval_list = get_txt_from_url(URL['eval_split'])
    check_in_eval = {e + '-05': True for e in eval_list}
    acc = [
        r[1] == r[2] for r in results
        if check_in_eval.get(r[0].split('/')[-1][:-4])
    ]
    acc = sum(acc) / len(acc)
    logger.warning(f'Zero-shot validation accuracy: {acc}')

    acc = [r[1] == r[2] for r in results]
    acc = sum(acc) / len(acc)
    logger.info(f'Zero-shot accuracy for the whole dataset: {acc}')
