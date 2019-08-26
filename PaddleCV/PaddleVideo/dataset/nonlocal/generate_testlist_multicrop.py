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

vallist = 'vallist.txt'
testlist = 'testlist.txt'
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
