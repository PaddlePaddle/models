import os
import numpy as np
import sys

num_classes = 400
replace_space_by_underliner = True  # whether to replace space by '_' in labels

fn = sys.argv[1]  #'trainlist_download400.txt'
train_dir = sys.argv[
    2]  #'/docker_mount/data/k400/Kinetics_trimmed_processed_train'
val_dir = sys.argv[3]  #'/docker_mount/data/k400/Kinetics_trimmed_processed_val'
trainlist = sys.argv[4]  #'trainlist.txt'
vallist = sys.argv[5]  #'vallist.txt'

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


generate_file(action_label_dict, train_dir, trainlist, num_classes)
generate_file(action_label_dict, val_dir, vallist, num_classes)
