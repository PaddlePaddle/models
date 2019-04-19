# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import numpy as np
import random

# src = 'trainlist_download.txt'
# outlist = 'trainlist.txt'
# original_folder = '/nfs.yoda/xiaolonw/kinetics/data/train'
# replace_folder = '/scratch/xiaolonw/kinetics/data/compress/train_256'
assert (len(sys.argv) == 5)

src = sys.argv[1]
outlist = sys.argv[2]
original_folder = sys.argv[3]
replace_folder = sys.argv[4]

f = open(src, 'r')
flist = []
for line in f:
    flist.append(line)
f.close()

f2 = open(outlist, 'w')

listlen = len(flist)
for i in range(listlen):
    line = flist[i]
    line = line.replace(original_folder, replace_folder)
    f2.write(line)

f2.close()
