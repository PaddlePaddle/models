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
"""
Split file into parts
"""
import sys
import os
block = int(sys.argv[1])
datadir = sys.argv[2]
file_list = []
for i in range(block):
    file_list.append(open(datadir + "/part-" + str(i), "w"))
id_ = 0
for line in sys.stdin:
    file_list[id_ % block].write(line)
    id_ += 1
for f in file_list:
    f.close()
