import os
import shutil

# set path
train_path = 'train/'
if not os.path.exists(train_path):
    os.makedirs(train_path)

test_path = 'test/'
if not os.path.exists(test_path):
    os.makedirs(test_path)

# move data
frame_dir = 'frame/'
f = open('ucfTrainTestlist/trainlist01.txt')
for line in f.readlines():
    folder = line.split('.')[0]
    vidid = folder.split('/')[-1]

    shutil.move(frame_dir + folder, train_path + vidid)
f.close()

f = open('ucfTrainTestlist/testlist01.txt')
for line in f.readlines():
    folder = line.split('.')[0]
    vidid = folder.split('/')[-1]

    shutil.move(frame_dir + folder, test_path + vidid)
f.close()
