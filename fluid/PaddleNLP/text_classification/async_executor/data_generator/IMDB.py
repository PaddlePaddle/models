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

import re
import os, sys
sys.path.append(os.path.abspath(os.path.join('..')))
from data_generator import MultiSlotDataGenerator


class IMDbDataGenerator(MultiSlotDataGenerator):
    def load_resource(self, dictfile):
        self._vocab = {}
        wid = 0
        with open(dictfile) as f:
            for line in f:
                self._vocab[line.strip()] = wid
                wid += 1
        self._unk_id = len(self._vocab)
        self._pattern = re.compile(r'(;|,|\.|\?|!|\s|\(|\))')

    def process(self, line):
        send = '|'.join(line.split('|')[:-1]).lower().replace("<br />",
                                                              " ").strip()
        label = [int(line.split('|')[-1])]

        words = [x for x in self._pattern.split(send) if x and x != " "]
        feas = [
            self._vocab[x] if x in self._vocab else self._unk_id for x in words
        ]

        return ("words", feas), ("label", label)


imdb = IMDbDataGenerator()
imdb.load_resource("aclImdb/imdb.vocab")

# data from files
file_names = os.listdir(sys.argv[1])
filelist = []
for i in range(0, len(file_names)):
    filelist.append(os.path.join(sys.argv[1], file_names[i]))

line_limit = 2500
process_num = 24
imdb.run_from_files(
    filelist=filelist,
    line_limit=line_limit,
    process_num=process_num,
    output_dir=('output_dataset/%s' % (sys.argv[1])))
