#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import argparse

parser = argparse.ArgumentParser(description='the direction of data list')
parser.add_argument(
    '--direction', type=str, default='B2A', help='the direction of data list')


def make_pair_data(fileA, file, d):
    f = open(fileA, 'r')
    lines = f.readlines()
    w = open(file, 'w')
    for line in lines:
        fileA = line[:-1]
        print(fileA)
        fileB = fileA.replace("A", "B")
        print(fileB)
        if d == 'A2B':
            l = fileA + '\t' + fileB + '\n'
        elif d == 'B2A':
            l = fileB + '\t' + fileA + '\n'
        else:
            raise NotImplementedError("the direction: [%s] is not support" % d)
        w.write(l)
    w.close()


if __name__ == "__main__":
    args = parser.parse_args()
    trainA_file = os.path.join("data", "cityscapes", "trainA.txt")
    train_file = os.path.join("data", "cityscapes", "pix2pix_train_list")
    make_pair_data(trainA_file, train_file, args.direction)

    testA_file = os.path.join("data", "cityscapes", "testA.txt")
    test_file = os.path.join("data", "cityscapes", "pix2pix_test_list")
    make_pair_data(testA_file, test_file, args.direction)
