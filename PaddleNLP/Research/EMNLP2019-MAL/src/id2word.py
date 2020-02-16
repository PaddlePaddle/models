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

import sys

id2word = {}
ln = sys.stdin

def load_vocab(file_path):
    start_index = 0
    f = open(file_path, 'r')

    for line in f:
        line = line.strip()
        id2word[start_index] = line
        start_index += 1
    f.close()

if __name__=="__main__":
    load_vocab(sys.argv[1])
    while True:
        line = ln.readline().strip()
        if not line:
            break

        split_res = line.split(" ")
        output_str = ""
        for item in split_res:
            output_str += id2word[int(item.strip())]
            output_str += " "
        output_str = output_str.strip()
        print output_str

