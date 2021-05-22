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

import numpy as np
import paddle
import paddle.nn.functional as F
from scipy import stats
from sklearn.metrics import average_precision_score

__all__ = [
    'save_checkpoint', 'load_checkpoint', 'get_labels', 'random_choice',
    'get_label_name_mapping'
]


def random_choice(a):
    i = np.random.randint(0, high=len(a), size=(1, ))
    return a[int(i)]


def get_labels():
    with open('./assets/audioset_labels.txt') as F:
        labels = F.read().split('\n')
    return labels


def get_ytid_clsidx_mapping():
    """
    Compute the mapping between youtube id and class index.
    The class index range from 0 to 527, correspoding to the labels stored in audioset_labels.txt file
    """
    labels = get_labels()
    label2clsidx = {l: i for i, l in enumerate(labels)}
    lines = open('./assets/unbalanced_train_segments.csv').read().split('\n')
    lines += open('./assets/balanced_train_segments.csv').read().split('\n')
    lines += open('./assets/eval_segments.csv').read().split('\n')
    lines = [l for l in lines if len(l) > 0 and l[0] != '#']
    ytid2clsidx = {}
    for l in lines:
        ytid = l.split(',')[0]
        labels = l.split(',')[3:]
        cls_idx = []
        for label in labels:
            label = label.replace('"', '').strip()
            cls_idx.append(label2clsidx[label])
        ytid2clsidx.update({ytid: cls_idx})
    clsidx2ytid = {i: [] for i in range(527)}
    for k in ytid2clsidx.keys():
        for v in ytid2clsidx[k]:
            clsidx2ytid[v] += [k]
    return ytid2clsidx, clsidx2ytid


def get_metrics(label, pred):
    a = label
    b = (pred > 0.5).astype('int32')
    eps = 1e-8
    tp = np.sum(b[a == 1])
    fp = np.sum(b[a == 0])
    precision = tp / (fp + tp + eps)
    fn = np.sum(b[a == 1] == 0)
    recall = tp / (tp + fn)

    return precision, recall


def compute_dprime(auc):
    """Compute d_prime metric.

    Reference:
    J. F. Gemmeke, D. P. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter, “Audio Set: An ontology and humanlabeled dataset for audio events,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017, pp. 776–780.

    """
    dp = stats.norm().ppf(auc) * np.sqrt(2.0)
    return dp


def save_checkpoint(model_dir, step, model, optimizer, prefix):
    print(f'checkpointing at step {step}')
    paddle.save(model.state_dict(),
                model_dir + '/{}_checkpoint{}.pdparams'.format(prefix, step))
    paddle.save(optimizer.state_dict(),
                model_dir + '/{}_checkpoint{}.pdopt'.format(prefix, step))


def load_checkpoint(model_dir, epoch, prefix):
    file = model_dir + '/{}_checkpoint_model{}.tar'.format(prefix, epoch)
    print('loading checkpoing ' + file)
    model_dict = paddle.load(model_dir +
                             '/{}_checkpoint{}.pdparams'.format(prefix, epoch))
    optim_dict = paddle.load(model_dir +
                             '/{}_checkpoint{}.pdopt'.format(prefix, epoch))
    return model_dict, optim_dict


def get_label_name_mapping():
    with open('./assets/ontology.json') as F:
        ontology = json.load(F)
    label2name = {o['id']: o['name'] for o in ontology}
    name2label = {o['name']: o['id'] for o in ontology}
    return label2name, name2label


def download_assets():
    os.makedirs('./assets/', exist_ok=True)
    urls = [
        'https://raw.githubusercontent.com/audioset/ontology/master/ontology.json',
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv',
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv',
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv'
    ]
    for url in urls:
        fname = './assets/' + url.split('/')[-1]
        if os.path.exists(fname):
            continue

        cmd = 'wget ' + url + ' -O ' + fname
        print(cmd)
        os.system(cmd)


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


def mixup_data(x, y, alpha=1.0):
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
    return mixed_x, mixed_target
