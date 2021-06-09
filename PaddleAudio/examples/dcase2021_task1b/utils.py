# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

import json
import os
import pickle

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.utils import download

__all__ = [
    'save_checkpoint', 'load_checkpoint', 'MixUpLoss', 'mixup_data',
    'get_txt_from_url', 'get_feature_from_url'
]


def get_txt_from_url(url):
    """Download and read text lines from url, remove empty lines if any.
    """
    file_path = download.get_weights_path_from_url(url)
    with open(file_path) as f:
        lines = f.read().split('\n')
    return [l for l in lines if len(l) > 0]


def get_pickle_results(url):
    weight = download.get_weights_path_from_url(url)
    with open(weight, 'rb') as f:
        results = pickle.load(f)
    return results


def get_feature_from_url(url):
    """Download and read features as numpy array from url.
    """
    file_path = download.get_weights_path_from_url(url)
    feature = np.load(file_path)
    return feature


def save_checkpoint(model_dir, step, model, optimizer, prefix):
    print(f'checkpointing at step {step}')
    paddle.save(model.state_dict(),
                model_dir + '/{}_checkpoint{}.pdparams'.format(prefix, step))
    paddle.save(optimizer.state_dict(),
                model_dir + '/{}_checkpoint{}.pdopt'.format(prefix, step))


def load_checkpoint(model_dir, epoch, prefix):
    file = model_dir + '/{}_checkpoint_model{}.tar'.format(prefix, epoch)
    print('loading checkpoing ' + file)
    model_dict = paddle.load(model_dir + '/{}_checkpoint{}.pdparams'.format(
        prefix, epoch))
    optim_dict = paddle.load(model_dir + '/{}_checkpoint{}.pdopt'.format(prefix,
                                                                         epoch))
    return model_dict, optim_dict


class MixUpLoss(paddle.nn.Layer):
    """Define the mixup loss used in training audioset.

    Reference:
    Zhang, Hongyi, et al. “Mixup: Beyond Empirical Risk Minimization.” International Conference on Learning Representations, 2017.
    """

    def __init__(self, criterion):
        super(MixUpLoss, self).__init__()
        self.criterion = criterion

    def forward(self, pred, mixup_target):
        assert type(mixup_target) in [
            tuple, list
        ] and len(mixup_target
                  ) == 3, 'mixup data should be tuple consists of (ya,yb,lamda)'
        ya, yb, lamda = mixup_target
        return lamda * self.criterion(pred, ya) \
                + (1 - lamda) * self.criterion(pred, yb)

    def extra_repr(self):
        return 'MixUpLoss with {}'.format(self.criterion)


def mixup_data(x, y, p=None, alpha=1.0):
    """Mix the input data and label using mixup strategy,  returns mixed inputs,
    pairs of targets, and lambda

    Reference:
    Zhang, Hongyi, et al. “Mixup: Beyond Empirical Risk Minimization.” International Conference on Learning Representations, 2017.

    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    index = paddle.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * paddle.index_select(x, index)
    y_a, y_b = y, paddle.index_select(y, index)
    mixed_target = (y_a, y_b, lam)
    if p is not None:
        mixed_p = lam * p + (1 - lam) * paddle.index_select(p, index)
        return mixed_x, mixed_p, mixed_target
    else:
        return mixed_x, None, mixed_target
