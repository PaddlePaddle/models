#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
################################################################################

"""
 Specify the brief poi_qac_personalized.py
"""
import os
import sys
import re
import time
import numpy as np
import random
import paddle.fluid as fluid

from datasets.base_dataset import BaseDataset

reload(sys)
sys.setdefaultencoding('gb18030')


base_rule = re.compile("[\1\2]")

class PoiQacPersonalized(BaseDataset):
    """
    PoiQacPersonalized dataset 
    """
    def __init__(self, flags):
        super(PoiQacPersonalized, self).__init__(flags)
        self.inited_dict = False

    def parse_context(self, inputs):
        """
        provide input context
        """

        """
        set inputs_kv: please set key as the same as layer.data.name

        notice:
        (1)
        If user defined "inputs key" is different from layer.data.name,
        the frame will rewrite "inputs key" with layer.data.name
        (2)
        The param "inputs" will be passed to user defined nets class through
        the nets class interface function : net(self, FLAGS, inputs), 
        """ 
        if self._flags.use_personal:
            #inputs['user_loc_geoid'] = fluid.layers.data(name="user_loc_geoid", shape=[40],
            #        dtype="int64", lod_level=0) #from clk poi
            #inputs['user_bound_geoid'] = fluid.layers.data(name="user_bound_geoid", shape=[40],
            #        dtype="int64", lod_level=0) #from clk poi
            #inputs['user_time_id'] = fluid.layers.data(name="user_time_geoid", shape=[1],
            #        dtype="int64", lod_level=1) #from clk poi
            inputs['user_clk_geoid'] = fluid.layers.data(name="user_clk_geoid", shape=[40],
                    dtype="int64", lod_level=0) #from clk poi
            inputs['user_tag_id'] = fluid.layers.data(name="user_tag_id", shape=[1],
                    dtype="int64", lod_level=1) #from clk poi
            inputs['user_resident_geoid'] = fluid.layers.data(name="user_resident_geoid", shape=[40],
                    dtype="int64", lod_level=0) #home, company
            inputs['user_navi_drive'] = fluid.layers.data(name="user_navi_drive", shape=[1],
                    dtype="int64", lod_level=0) #driver or not
        
        inputs['prefix_letter_id'] = fluid.layers.data(name="prefix_letter_id", shape=[1],
                dtype="int64", lod_level=1)
        if self._flags.prefix_word_id:
            inputs['prefix_word_id'] = fluid.layers.data(name="prefix_word_id", shape=[1],
                dtype="int64", lod_level=1)
        inputs['prefix_loc_geoid'] = fluid.layers.data(name="prefix_loc_geoid", shape=[40],
                dtype="int64", lod_level=0)
        if self._flags.use_personal:
            inputs['prefix_time_id'] = fluid.layers.data(name="prefix_time_id", shape=[1],
                dtype="int64", lod_level=1)

        inputs['pos_name_letter_id'] = fluid.layers.data(name="pos_name_letter_id", shape=[1],
                dtype="int64", lod_level=1)
        inputs['pos_name_word_id'] = fluid.layers.data(name="pos_name_word_id", shape=[1],
                dtype="int64", lod_level=1)
        inputs['pos_addr_letter_id'] = fluid.layers.data(name="pos_addr_letter_id", shape=[1],
                dtype="int64", lod_level=1)
        inputs['pos_addr_word_id'] = fluid.layers.data(name="pos_addr_word_id", shape=[1],
                dtype="int64", lod_level=1)
        inputs['pos_loc_geoid'] = fluid.layers.data(name="pos_loc_geoid", shape=[40],
                dtype="int64", lod_level=0)
        if self._flags.use_personal:
            inputs['pos_tag_id'] = fluid.layers.data(name="pos_tag_id", shape=[1],
                dtype="int64", lod_level=1)

        if self.is_training:
            inputs['neg_name_letter_id'] = fluid.layers.data(name="neg_name_letter_id", shape=[1],
                    dtype="int64", lod_level=1)
            inputs['neg_name_word_id'] = fluid.layers.data(name="neg_name_word_id", shape=[1],
                    dtype="int64", lod_level=1)
            inputs['neg_addr_letter_id'] = fluid.layers.data(name="neg_addr_letter_id", shape=[1],
                    dtype="int64", lod_level=1)
            inputs['neg_addr_word_id'] = fluid.layers.data(name="neg_addr_word_id", shape=[1],
                    dtype="int64", lod_level=1)
            inputs['neg_loc_geoid'] = fluid.layers.data(name="neg_loc_geoid", shape=[40],
                    dtype="int64", lod_level=0)
            if self._flags.use_personal:
                inputs['neg_tag_id'] = fluid.layers.data(name="neg_tag_id", shape=[1],
                    dtype="int64", lod_level=1)
        else:
            #for predict label
            inputs['label'] = fluid.layers.data(name="label", shape=[1],
                dtype="int64", lod_level=0)

        context = {"inputs": inputs}

        #set debug list, print info during training
        #debug_list = [key for key in inputs]
        #context["debug_list"] = ["prefix_ids", "label"]

        return context

    def _init_dict(self):
        """
            init dict
        """
        if self.inited_dict:
            return
        
        if self._flags.platform in ('local-gpu', 'pserver-gpu', 'slurm'):
            gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
            self.place = fluid.CUDAPlace(gpu_id)
        else:
            self.place = fluid.CPUPlace()

        self.term_dict = {}
        if self._flags.qac_dict_path is not None:
            with open(self._flags.qac_dict_path, 'r') as f:
                for line in f:
                    term, term_id = line.strip('\r\n').split('\t')
                    self.term_dict[term] = int(term_id)

        self.tag_info = {}
        if self._flags.tag_dict_path is not None:
            with open(self._flags.tag_dict_path, 'r') as f:
                for line in f:
                    tag, level, tid  = line.strip('\r\n').split('\t')
                    self.tag_info[tag] =  map(int, tid.split(','))

        self.user_kv = None
        self.poi_kv = None 
        if self._flags.kv_path is not None:
            self.poi_kv = {}
            with open(self._flags.kv_path + "/sug_raw.dat", "r") as f:
                for line in f:
                    pid, val = line.strip('\r\n').split('\t', 1)
                   self.poi_kv[pid] = val

            self.user_kv = {}
            with open(self._flags.kv_path + "/user_profile.dat", "r") as f:
                for line in f:
                    uid, val = line.strip('\r\n').split('\t', 1)
                    self.user_kv[uid] = val

            sys.stderr.write("load user kv:%s\n" % self._flags.kv_path)

        self.inited_dict = True
        sys.stderr.write("loaded term dict:%s, tag_dict:%s\n" % (len(self.term_dict), len(self.tag_info)))

    def _get_time_id(self, ts):
        """
        get time id:0-27
        """
        ts_struct = time.localtime(ts)

        week = ts_struct[6]
        hour = ts_struct[3]

        base = 0
        if hour >= 0 and hour < 6:
            base = 0
        elif hour >= 6 and hour < 12:
            base = 1
        elif hour >= 12 and hour < 18:
            base = 2
        else:
            base = 3

        final = week * 4 + base
        return final

    def _pad_batch_data(self, insts, pad_idx, return_max_len=True, return_num_token=False):
        """
        Pad the instances to the max sequence length in batch, and generate the
        corresponding position data and attention bias.
        """
        return_list = []
        max_len = max(len(inst) for inst in insts)
        # Any token included in dict can be used to pad, since the paddings' loss
        # will be masked out by weights and make no effect on parameter gradients.
        inst_data = np.array(
            [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
        return_list += [inst_data.astype("int64").reshape([-1, 1])]
        
        if return_max_len:
            return_list += [max_len]
        if return_num_token:
            num_token = 0
            for inst in insts:
                num_token += len(inst)
            return_list += [num_token]
        return return_list if len(return_list) > 1 else return_list[0]

    def _get_tagid(self, tag_str):
        if len(tag_str.strip()) < 1:
            return []
        tags = set()
        for t in tag_str.split():
            if ':' in t: 
                t = t.split(':')[0]
            t = t.lower()
            if t in self.tag_info:
                tags.update(self.tag_info[t])
        return list(tags) 

    def _get_ids(self, seg_info):
        #phraseseg, basicseg = seg_info
         
        if len(seg_info) < 2:
            return [0], [0]
        _, bt = [x.split('\3') for x in seg_info]

        rq = "".join(bt)
        bl = [t.encode('gb18030') for t in rq.decode('gb18030')]
        letter_ids = [] 
        for t in bl:
            letter_ids.append(self.term_dict.get(t.lower(), 1))
            if len(letter_ids) >= self._flags.max_seq_len:
                break

        word_ids = []
        for t in bt:
            word_ids.append(self.term_dict.get(t.lower(), 1)) 
            if len(word_ids) >= self._flags.max_seq_len:
                break
        return letter_ids, word_ids
 
    def _get_poi_ids(self, poi_str, max_num=0):
        if len(poi_str) < 1:
            return []
        ids = []
        all_p = poi_str.split('\1')
        
        pidx = range(0, len(all_p))
        if max_num > 0:
            #neg sample: last 10 is negative sampling
            if len(all_p) > max_num:
                neg_s_idx = len(all_p) - 10
                pidx = [1, 2] + random.sample(pidx[3:neg_s_idx], max_num - 13) + pidx[neg_s_idx:] 
            else:
                pidx = pidx[1:]
        bids = set()
        for x in pidx:
            poi_seg = all_p[x].split('\2')
            tagid = [0] 
            if len(poi_seg) >= 9:
                #name, uid, index, name_lid, name_wid, addr_lid, addr_wid, geohash, tagid
                bid = poi_seg[1]
                name_letter_id = map(int, poi_seg[3].split())[:self._flags.max_seq_len]
                name_word_id = map(int, poi_seg[4].split())[:self._flags.max_seq_len]
                addr_letter_id = map(int, poi_seg[5].split())[:self._flags.max_seq_len]
                addr_word_id = map(int, poi_seg[6].split())[:self._flags.max_seq_len]
                ghid = map(int, poi_seg[7].split(','))
                if len(poi_seg[8]) > 0:
                    tagid = map(int, poi_seg[8].split(','))
            else:
                #raw_text: uid, name, addr, xy, tag, alias
                bid = poi_seg[0]
                name_letter_id, name_word_id = self._get_ids(poi_seg[1])
                addr_letter_id, addr_word_id = self._get_ids(poi_seg[2])
                ghid = map(int, poi_seg[3].split(',')) 
                if len(poi_seg[4]) > 0:
                    tagid = map(int, poi_seg[4].split(','))

            if not self.is_training and name_letter_id == [0]:
                continue # empty name
            if bid in bids:
                continue
            bids.add(bid)
            ids.append([name_letter_id, name_word_id, addr_letter_id, addr_word_id, ghid, tagid])

        return ids

    def _get_user_ids(self, cuid, user_str):
        if self.user_kv:
            if cuid in self.user_kv:
                val = self.user_kv[cuid]
                drive_conf, clk_p, res_p = val.split('\t')
            else:
                return []
        else:
            if len(user_str) < 1:
                return []
            drive_conf, clk_p, res_p = user_str.split('\1')
            
        ids = []
        conf1, conf2 = drive_conf.split('\2')
        is_driver = 0
        if float(conf1) > 0.5 or float(conf2) > 1.5:
            is_driver = 1
        
        user_clk_geoid = [0] * 40
        user_tag_id = set()
        if len(clk_p) > 0:
            if self.user_kv:
                for p in clk_p.split('\1'):
                    bid, time, loc, bound = p.split('\2')
                    if bid in self.poi_kv:
                        v = self.poi_kv[bid]
                        v = base_rule.sub("", v)
                        info = v.split('\t') #name, addr, ghid, tag, alias
                        ghid = map(int, info[2].split(',')) 
                        for i in range(len(user_clk_geoid)):
                            user_clk_geoid[i] = user_clk_geoid[i] | ghid[i]
                        user_tag_id.update(self._get_tagid(info[4]))
            else:
                for p in clk_p.split('\2'):
                    bid, gh, tags = p.split('\3')
                    ghid = map(int, gh.split(',')) 
                    for i in range(len(user_clk_geoid)):
                        user_clk_geoid[i] = user_clk_geoid[i] | ghid[i]
                    if len(tags) > 0:
                        user_tag_id.update(tags.split(','))
        if len(user_tag_id) < 1:
            user_tag_id = [0]
        user_tag_id = map(int, list(user_tag_id))
        ids.append(user_clk_geoid)
        ids.append(user_tag_id)

        user_res_geoid = [0] * 40
        if len(res_p) > 0:
            if self.user_kv:
                for p in res_p.split('\1'):
                    bid, conf = p.split('\2')
                    if bid in self.poi_kv:
                        v = self.poi_kv[bid]
                        v = base_rule.sub("", v)
                        info = v.split('\t') #name, addr, ghid, tag, alias
                        ghid = map(int, info[2].split(','))
                        for i in range(len(user_res_geoid)):
                            user_res_geoid[i] = user_res_geoid[i] | ghid[i]
            else:
                for p in res_p.split('\2'):
                    bid, gh, conf = p.split('\3')
                    ghid = map(int, gh.split(','))
                    for i in range(len(user_res_geoid)):
                        user_res_geoid[i] = user_res_geoid[i] | ghid[i]
        ids.append(user_res_geoid)
        ids.append([is_driver])
        return ids

    def parse_batch(self, data_gen):
        """
        reader_batch must be true: only for train & loss_func is log_exp, other use parse_oneline
        pos : neg = 1 : N
        """
        batch_data = {}
        def _get_lod(k):
            #sys.stderr.write("%s\t%s\t%s\n" % (k, " ".join(map(str, batch_data[k][0])),
            #            " ".join(map(str, batch_data[k][1])) ))
            return fluid.create_lod_tensor(np.array(batch_data[k][0]).reshape([-1, 1]),
                    [batch_data[k][1]], self.place)
        
        keys = None
        for line in data_gen():
            for s in self.parse_oneline(line):
                for k, v in s:
                    if k not in batch_data:
                        batch_data[k] = [[], []]

                    if not isinstance(v[0], list):
                        v = [v] #pos 1 to N
                    for j in v:
                        batch_data[k][0].extend(j)
                        batch_data[k][1].append(len(j))

                if keys is None:
                    keys = [k for k, _ in s]
                if len(batch_data[keys[0]][1]) == self._flags.batch_size:
                    yield [(k, _get_lod(k)) for k in keys]
                    batch_data = {}
        
        if not self._flags.drop_last_batch and len(batch_data) != 0:
            yield [(k, _get_lod(k)) for k in keys]

    def parse_oneline(self, line):
        """
        datareader interface
        """
        self._init_dict()

        qid, user, prefix, pos_poi, neg_poi = line.strip("\r\n").split("\t")
        cuid, time, loc_cityid, bound_cityid, loc_gh, bound_gh = qid.split('_') 
       
        #step1
        user_input = []
        if self._flags.use_personal:
            user_ids = self._get_user_ids(cuid, user)
            if len(user_ids) < 1:
                user_ids = [[0] * 40, [0], [0] * 40, [0]]
            user_input = [("user_clk_geoid", user_ids[0]), \
                          ("user_tag_id", user_ids[1]), \
                          ("user_resident_geoid", user_ids[2]), \
                          ("user_navi_drive", user_ids[3])]

        #step2
        prefix_seg = prefix.split('\2')
        prefix_time_id = self._get_time_id(int(time)) 
        prefix_loc_geoid = [0] * 40 
        if len(prefix_seg) >= 4: #query, letterid, wordid, ghid, poslen, neglen
            prefix_letter_id = map(int, prefix_seg[1].split())[:self._flags.max_seq_len]
            prefix_word_id = map(int, prefix_seg[2].split())[:self._flags.max_seq_len]
            loc_gh, bound_gh = prefix_seg[3].split('_')
            ghid = map(int, loc_gh.split(','))
            for i in range(len(prefix_loc_geoid)):
                prefix_loc_geoid[i] = prefix_loc_geoid[i] | ghid[i]
            ghid = map(int, bound_gh.split(','))
            for i in range(len(prefix_loc_geoid)):
                prefix_loc_geoid[i] = prefix_loc_geoid[i] | ghid[i]
        else: #raw text
            prefix_letter_id, prefix_word_id = self._get_ids(prefix)
            ghid = map(int, loc_gh.split(','))
            for i in range(len(prefix_loc_geoid)):
                prefix_loc_geoid[i] = prefix_loc_geoid[i] | ghid[i]
            ghid = map(int, bound_gh.split(','))
            for i in range(len(prefix_loc_geoid)):
                prefix_loc_geoid[i] = prefix_loc_geoid[i] | ghid[i]

        prefix_input = [("prefix_letter_id", prefix_letter_id), \
                    ("prefix_loc_geoid", prefix_loc_geoid)]

        if self._flags.prefix_word_id:
            prefix_input.insert(1, ("prefix_word_id", prefix_word_id))

        if self._flags.use_personal:
            prefix_input.append(("prefix_time_id", [prefix_time_id]))

        #step3
        pos_ids = self._get_poi_ids(pos_poi)
        pos_num = len(pos_ids)
        max_num = 0
        if self.is_training:
            max_num = max(20, self._flags.neg_sample_num) #last 10 is neg sample
        neg_ids = self._get_poi_ids(neg_poi, max_num=max_num)
        #if not train, add all pois
        if not self.is_training:
            pos_ids.extend(neg_ids)
            if len(pos_ids) < 1:
                pos_ids.append([[0], [0], [0], [0], [0] * 40, [0]])

        #step4
        idx = 0
        for pos_id in pos_ids:
            pos_input = [("pos_name_letter_id", pos_id[0]), \
                        ("pos_name_word_id", pos_id[1]), \
                        ("pos_addr_letter_id", pos_id[2]), \
                        ("pos_addr_word_id", pos_id[3]), \
                        ("pos_loc_geoid", pos_id[4])]

            if self._flags.use_personal:
                pos_input.append(("pos_tag_id", pos_id[5]))

            if self.is_training:
                if len(neg_ids) > self._flags.neg_sample_num:
                    #Noise Contrastive Estimation
                    #if self._flags.neg_sample_num > 3:
                    #    nids_sample = neg_ids[:3]
                    nids_sample = random.sample(neg_ids, self._flags.neg_sample_num)
                else:
                    nids_sample = neg_ids

                if self._flags.reader_batch:
                    if len(nids_sample) != self._flags.neg_sample_num:
                        continue

                    neg_batch = [[], [], [], [], [], []]
                    for neg_id in nids_sample:
                        for i in range(len(neg_batch)):
                            neg_batch[i].append(neg_id[i]) 
                    
                    neg_input = [("neg_name_letter_id", neg_batch[0]), \
                                ("neg_name_word_id", neg_batch[1]), \
                                ("neg_addr_letter_id", neg_batch[2]), \
                                ("neg_addr_word_id", neg_batch[3]), \
                                ("neg_loc_geoid", neg_batch[4])]
                    if self._flags.use_personal:
                        neg_input.append(("neg_tag_id", neg_batch[5]))
                    yield user_input + prefix_input + pos_input + neg_input
                else:
                    for neg_id in nids_sample:
                        neg_input = [("neg_name_letter_id", neg_id[0]), \
                                    ("neg_name_word_id", neg_id[1]), \
                                    ("neg_addr_letter_id", neg_id[2]), \
                                    ("neg_addr_word_id", neg_id[3]), \
                                    ("neg_loc_geoid", neg_id[4])]
                        if self._flags.use_personal:
                            neg_input.append(("neg_tag_id", neg_id[5]))
                        yield user_input + prefix_input + pos_input + neg_input
            else:
                label = int(idx < pos_num)
                yield user_input + prefix_input + pos_input + [("label", [label])]

            idx += 1


if __name__ == '__main__':
    from utils import flags
    from utils.load_conf_file import LoadConfFile
    FLAGS = flags.FLAGS
    flags.DEFINE_custom("conf_file", "./conf/test/test.conf", 
        "conf file", action=LoadConfFile, sec_name="Train")
    
    sys.stderr.write('-----------  Configuration Arguments -----------\n')
    for arg, value in sorted(flags.get_flags_dict().items()):
        sys.stderr.write('%s: %s\n' % (arg, value))
    sys.stderr.write('------------------------------------------------\n')
   
    dataset_instance = PoiQacPersonalized(FLAGS)
    def _dump_vec(data, name):
        print("%s\t%s" % (name, " ".join(map(str, np.array(data)))))
    
    def _data_generator(): 
        """
        stdin sample generator: read from stdin 
        """
        for line in sys.stdin:
            if not line.strip():
                continue
            yield line

    if FLAGS.reader_batch: 
        for sample in dataset_instance.parse_batch(_data_generator):
            _dump_vec(sample[0][1], 'user_clk_geoid')
            _dump_vec(sample[1][1], 'user_tag_id')
            _dump_vec(sample[2][1], 'user_resident_geoid')
            _dump_vec(sample[3][1], 'user_navi_drive')
            _dump_vec(sample[4][1], 'prefix_letter_id')
            _dump_vec(sample[5][1], 'prefix_loc_geoid')
            _dump_vec(sample[6][1], 'prefix_time_id')
            _dump_vec(sample[7][1], 'pos_name_letter_id')
            _dump_vec(sample[10][1], 'pos_addr_word_id')
            _dump_vec(sample[11][1], 'pos_loc_geoid')
            _dump_vec(sample[12][1], 'pos_tag_id')
            _dump_vec(sample[13][1], 'neg_name_letter_id or label')
    else:
        for line in sys.stdin:
            for sample in dataset_instance.parse_oneline(line):
                _dump_vec(sample[0][1], 'user_clk_geoid')
                _dump_vec(sample[1][1], 'user_tag_id')
                _dump_vec(sample[2][1], 'user_resident_geoid')
                _dump_vec(sample[3][1], 'user_navi_drive')
                _dump_vec(sample[4][1], 'prefix_letter_id')
                _dump_vec(sample[5][1], 'prefix_loc_geoid')
                _dump_vec(sample[6][1], 'prefix_time_id')
                _dump_vec(sample[7][1], 'pos_name_letter_id')
                _dump_vec(sample[10][1], 'pos_addr_word_id')
                _dump_vec(sample[11][1], 'pos_loc_geoid')
                _dump_vec(sample[12][1], 'pos_tag_id')
                _dump_vec(sample[13][1], 'neg_name_letter_id or label')

