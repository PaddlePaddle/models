#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import os
import numpy as np
import sys

num_classes = 400
replace_space_by_underliner = True  # whether to replace space by '_' in labels
train_dir = sys.argv[
    1]  #e.g., '/docker_mount/data/k400/Kinetics_trimmed_processed_train'
val_dir = sys.argv[
    2]  #e.g., '/docker_mount/data/k400/Kinetics_trimmed_processed_val'
fn = 'kinetics-400_train.csv'  # this should be download first from ActivityNet
trainlist = 'trainlist.txt'
vallist = 'vallist.txt'
testlist = 'testlist.txt'

fl = open(fn).readlines()
fl = [line.strip() for line in fl if line.strip() != '']
action_list = []

for line in fl[1:]:
    act = line.split(',')[0].strip('\"')
    action_list.append(act)

action_set = set(action_list)
action_list = list(action_set)
action_list.sort()
if replace_space_by_underliner:
    action_list = [item.replace(' ', '_') for item in action_list]

# assign integer label to each category, abseiling is labeled as 0, 
# zumba labeled as 399 and so on, sorted by the category name
action_label_dict = {}
for i in range(len(action_list)):
    key = action_list[i]
    action_label_dict[key] = i

assert len(action_label_dict.keys(
)) == num_classes, "action num should be {}".format(num_classes)


def generate_file(Faction_label_dict, Ftrain_dir, Ftrainlist, Fnum_classes):
    trainactions = os.listdir(Ftrain_dir)
    trainactions.sort()
    assert len(
        trainactions) == Fnum_classes, "train action num should be {}".format(
            Fnum_classes)

    train_items = []
    trainlist_outfile = open(Ftrainlist, 'w')
    for trainaction in trainactions:
        assert trainaction in Faction_label_dict.keys(
        ), "action {} should be in action_dict".format(trainaction)
        trainaction_dir = os.path.join(Ftrain_dir, trainaction)
        trainaction_label = Faction_label_dict[trainaction]
        trainaction_files = os.listdir(trainaction_dir)
        for f in trainaction_files:
            fn = os.path.join(trainaction_dir, f)
            item = fn + ' ' + str(trainaction_label)
            train_items.append(item)
            trainlist_outfile.write(item + '\n')
    trainlist_outfile.flush()
    trainlist_outfile.close()


###### generate file list for training
generate_file(action_label_dict, train_dir, trainlist, num_classes)
###### generate file list for validation
generate_file(action_label_dict, val_dir, vallist, num_classes)

###### generate file list for evaluation
sampling_times = 10
cropping_times = 3

fl = open(vallist).readlines()
fl = [line.strip() for line in fl if line.strip() != '']
f_test = open(testlist, 'w')

for i in range(len(fl)):
    line = fl[i].split(' ')
    fn = line[0]
    label = line[1]
    for j in range(sampling_times):
        for k in range(cropping_times):
            test_item = fn + ' ' + str(i) + ' ' + str(j) + ' ' + str(k) + '\n'
            f_test.write(test_item)

f_test.close()
