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
from paddleaudio.utils.log import logger
from sklearn.linear_model import LogisticRegression
from utils import get_feature_from_url, get_txt_from_url

URL = {
    'train_feat':
    'https://bj.bcebos.com/paddleaudio/examples/dcase21_task1b/clip_visual_lp_train_features.npy',
    'eval_feat':
    'https://bj.bcebos.com/paddleaudio/examples/dcase21_task1b/clip_visual_lp_val_features.npy',
    'image_list':
    'https://bj.bcebos.com/paddleaudio/examples/dcase21_task1b/image_list.txt',
    'train_split':
    'https://bj.bcebos.com/paddleaudio/examples/dcase21_task1b/train_split.txt',
    'eval_split':
    'https://bj.bcebos.com/paddleaudio/examples/dcase21_task1b/eval_split.txt'
}


def split_dataset(image_files, train_split):
    """Split the dataset as described in the dcase 2021 task1b baseline
    Reference:
    https://github.com/marmoi/dcase2021_task1a_baseline

    """
    train_files = []
    val_files = []
    train_split = {t: True for t in train_split}
    for f in image_files:
        key = f.split('/')[-1][:-7]

        if train_split.get(key, False):
            train_files += [f]
        else:
            val_files += [f]
    return train_files, val_files


if __name__ == '__main__':
    image_files = get_txt_from_url(URL['image_list'])
    train_split = get_txt_from_url(URL['train_split'])
    eval_split = get_txt_from_url(URL['eval_split'])

    train_files, val_files = split_dataset(image_files, train_split)
    # get labels directly from filenames
    train_labels = [
        f.split('/')[-1].split('-')[0] for f in train_files if len(f) > 0
    ]
    eval_labels = [
        f.split('/')[-1].split('-')[0] for f in val_files if len(f) > 0
    ]

    logger.info(
        f'train files {len(train_files)}, validation files: {len(val_files)}')

    train_features = get_feature_from_url(URL['train_feat'])
    eval_features = get_feature_from_url(URL['eval_feat'])

    logger.info('Training logistic regression...')
    # train the logistic regression model
    classifier = LogisticRegression(
        random_state=0, C=0.01, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)
    logger.info('done')
    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(eval_features)
    accuracy = np.mean((eval_labels == predictions).astype(np.float)) * 100.
    logger.info(f"Accuracy = {accuracy:.3f}")
