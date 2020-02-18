#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

# select sentence vector and featmap of one movie name for inference
import os
import sys
import pickle
import numpy as np

infer_path = 'infer'
infer_feat_path = 'infer/infer_feat'

if not os.path.exists(infer_path):
    os.mkdir(infer_path)
if not os.path.exists(infer_feat_path):
    os.mkdir(infer_feat_path)

python_ver = sys.version_info

pickle_path = 'test_clip-sentvec.pkl'
if python_ver < (3, 0):
    movies_sentence = pickle.load(open(pickle_path, 'rb'))
else:
    movies_sentence = pickle.load(open(pickle_path, 'rb'), encoding='bytes')

select_name = movies_sentence[0][0].split('.')[0]

res_sentence = []
for movie_sentence in movies_sentence:
    if movie_sentence[0].split('.')[0] == select_name:
        res_sen = []
        res_sen.append(movie_sentence[0])
        res_sen.append([movie_sentence[1][0]])  #select the first one sentence
        res_sentence.append(res_sen)

file = open('infer/infer_clip-sen.pkl', 'wb')
pickle.dump(res_sentence, file, protocol=2)

movies_feat = os.listdir('Interval128_256_overlap0.8_c3d_fc6')
for movie_feat in movies_feat:
    if movie_feat.split('.')[0] == select_name:
        feat_path = os.path.join('Interval128_256_overlap0.8_c3d_fc6',
                                 movie_feat)
        feat = np.load(feat_path)
        np.save(os.path.join(infer_feat_path, movie_feat), feat)
