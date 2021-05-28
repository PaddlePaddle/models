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

import glob

import numpy as np
import paddle
import paddle.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from utils import (build_rn50_model, build_rn101_model, build_vit_model,
                   tokenize, transform)

image_files = glob.glob('images/*.png')
# use only the middle frame of a video
image_files = [f for f in image_files if '-05' in f]
train_split_file = '<train_split.txt>'
eval_split_file = '<eval_split.txt>'


def split_dataset(image_files, train_split_file):
    """split the dataset as described in the dcase baseline"""
    train_files = []
    val_files = []
    train_split = {t[:-1]: True for t in open(train_split_file).readlines()}
    for f in image_files:
        key = f.split('/')[-1][:-7]

        if train_split.get(key, False):
            train_files += [f]
        else:
            val_files += [f]
    return train_files, val_files


def get_features(img_files):
    features = []
    for f in tqdm(img_files):
        img = Image.open(f)
        image_input = transform(img)
        image_feature = model.encode_image(image_input).numpy()
        features += [image_feature]
    return np.concatenate(features, 0)


train_files, val_files = split_dataset(image_files, train_split_file)
print(f'train files {len(train_files)}, validation files: {len(val_files)}')

# build model and load the pre-trained weight.
model = build_vit_model()
sd = paddle.load('./assets/ViT-B-32.pdparams')
model.load_dict(sd)
model.eval()

# compute features using clip
train_features = get_features(train_files)
val_features = get_features(val_files)
# get labels directly from filenames
train_labels = [f.split('/')[-1].split('-')[0] for f in train_files]
val_labels = [f.split('/')[-1].split('-')[0] for f in val_files]

# train the logistic regression model
classifier = LogisticRegression(random_state=0,
                                C=0.316,
                                max_iter=1000,
                                verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(val_features)
accuracy = np.mean((val_labels == predictions).astype(np.float)) * 100.
print(f"Accuracy = {accuracy:.3f}")
