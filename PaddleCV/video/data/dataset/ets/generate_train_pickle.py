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

import pickle
import os

import multiprocessing

output_dir = './feat_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
fname = 'resnet152_features_activitynet_5fps_320x240.pkl'
d = pickle.load(open(fname))


def save_file(filenames, process_id):
    count = 0
    for key in filenames:
        pickle.dump(d[key], open(os.path.join(output_dir, key), 'w'))
        count += 1
        if count % 100 == 0:
            print('# %d processed %d samples' % (process_id, count))
    print('# %d total processed %d samples' % (process_id, count))


total_keys = d.keys()

num_threads = 8
filelists = [None] * 8
seg_nums = len(total_keys) // 8

p_list = [None] * 8

for i in range(8):
    if i == 7:
        filelists[i] = total_keys[i * seg_nums:]
    else:
        filelists[i] = total_keys[i * seg_nums:(i + 1) * seg_nums]

    p_list[i] = multiprocessing.Process(
        target=save_file, args=(filelists[i], i))

    p_list[i].start()
