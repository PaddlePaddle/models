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
import json
import pandas as pd

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: tojson.py <input_path> <output_path>')
        exit()
    infile = sys.argv[1]
    outfile = sys.argv[2]
    df = pd.read_json(infile)
    with open(outfile, 'w') as f:
        for row in df.iterrows():
            f.write(row[1].to_json() + '\n')
