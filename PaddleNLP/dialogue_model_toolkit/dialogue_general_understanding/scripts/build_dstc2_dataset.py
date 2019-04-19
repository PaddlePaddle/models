# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""build mrda train dev test dataset"""

import json
import sys
import csv
import os
import re

import commonlib


class DSTC2(object): 
    """
    dialogue state tracking dstc2 data process
    """
    def __init__(self): 
        """
        init instance
        """
        self.map_tag_dict = {}
        self.out_dir = "../data/dstc2/dstc2"
        self.out_asr_dir = "../data/dstc2/dstc2_asr"
        self.data_list = "./conf/dstc2.conf"
        self.map_tag = "../data/dstc2/dstc2/map_tag_id.txt"
        self.src_dir = "../data/dstc2/source_data"
        self.onto_json = "../data/dstc2/source_data/ontology_dstc2.json"
        self._load_file()
        self._load_ontology()

    def _load_file(self): 
        """
        load dataset filename
        """
        self.data_dict = commonlib.load_dict(self.data_list)
        for data_type in self.data_dict: 
            for i in range(len(self.data_dict[data_type])): 
                self.data_dict[data_type][i] = os.path.join(self.src_dir, self.data_dict[data_type][i])

    def _load_ontology(self): 
        """
        load ontology tag
        """
        tag_id = 1
        self.map_tag_dict['none'] = 0
        with open(self.onto_json, 'r') as fr: 
            ontology = json.load(fr)
            slots_values = ontology['informable']
            for slot in slots_values: 
                for value in slots_values[slot]: 
                    key = "%s_%s" % (slot, value)
                    self.map_tag_dict[key] = tag_id
                    tag_id += 1
                key = "%s_none" % (slot)
                self.map_tag_dict[key] = tag_id
                tag_id += 1

    def _parser_dataset(self, data_type): 
        """
        parser train dev test dataset
        """
        stat = os.path.exists(self.out_dir)
        if not stat: 
            os.makedirs(self.out_dir)
        asr_stat = os.path.exists(self.out_asr_dir)
        if not asr_stat: 
            os.makedirs(self.out_asr_dir)
        out_file = os.path.join(self.out_dir, "%s.txt" % data_type)
        out_asr_file = os.path.join(self.out_asr_dir, "%s.txt" % data_type)
        with open(out_file, 'w') as fw, open(out_asr_file, 'w') as fw_asr: 
            data_list = self.data_dict.get(data_type)
            for fn in data_list: 
                log_file = os.path.join(fn, "log.json")
                label_file = os.path.join(fn, "label.json")
                with open(log_file, 'r') as f_log, open(label_file, 'r') as f_label: 
                    log_json = json.load(f_log)
                    label_json = json.load(f_label)
                    session_id = log_json['session-id']
                    assert len(label_json["turns"]) == len(log_json["turns"])
                    for i in range(len(label_json["turns"])): 
                        log_turn = log_json["turns"][i]
                        label_turn = label_json["turns"][i]
                        assert log_turn["turn-index"] == label_turn["turn-index"]
                        labels = ["%s_%s" % (slot, label_turn["goal-labels"][slot]) for slot in label_turn["goal-labels"]]
                        labels_ids = " ".join([str(self.map_tag_dict.get(label, self.map_tag_dict["%s_none" % label.split('_')[0]])) for label in labels])
                        mach = log_turn['output']['transcript']
                        user = label_turn['transcription']
                        if not labels_ids.strip(): 
                            labels_ids = self.map_tag_dict['none']
                        out = "%s\t%s\1%s\t%s" % (session_id, mach, user, labels_ids)
                        user_asr = log_turn['input']['live']['asr-hyps'][0]['asr-hyp'].strip()
                        out_asr = "%s\t%s\1%s\t%s" % (session_id, mach, user_asr, labels_ids)
                        fw.write("%s\n" % out.encode('utf8'))
                        fw_asr.write("%s\n" % out_asr.encode('utf8'))

    def get_train_dataset(self): 
        """
        parser train dataset and print train.txt
        """
        self._parser_dataset("train")

    def get_dev_dataset(self): 
        """
        parser dev dataset and print dev.txt
        """
        self._parser_dataset("dev")

    def get_test_dataset(self): 
        """
        parser test dataset and print test.txt
        """
        self._parser_dataset("test")

    def get_labels(self): 
        """
        get tag and map ids file
        """
        with open(self.map_tag, 'w') as fw: 
            for elem in self.map_tag_dict: 
                fw.write("%s\t%s\n" % (elem, self.map_tag_dict[elem]))

    def main(self): 
        """
        run data process
        """
        self.get_train_dataset()
        self.get_dev_dataset()
        self.get_test_dataset()
        self.get_labels()

if __name__ == "__main__": 
    dstc_inst = DSTC2()
    dstc_inst.main()




