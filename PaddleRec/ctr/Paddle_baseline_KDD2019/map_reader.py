# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
import paddle.fluid.incubate.data_generator as dg


class MapDataset(dg.MultiSlotDataGenerator):
    def setup(self, sparse_feature_dim):
        self.profile_length = 65
        self.dense_length = 3
        #feature names
        self.dense_feature_list = ["distance", "price", "eta"]

        self.pid_list = ["pid"]
        self.query_feature_list = ["weekday", "hour", "o1", "o2", "d1", "d2"]
        self.plan_feature_list = ["transport_mode"]
        self.rank_feature_list = ["plan_rank", "whole_rank", "price_rank", "eta_rank", "distance_rank"]
        self.rank_whole_pic_list = ["mode_rank1", "mode_rank2", "mode_rank3", "mode_rank4",
                                    "mode_rank5"]
        self.weather_feature_list = ["max_temp", "min_temp", "wea", "wind"]
        self.hash_dim = 1000001
        self.train_idx_ = 2000000
        #carefully set if you change the features 
        self.categorical_range_ = range(0, 22)

    #process one instance
    def _process_line(self, line):
        instance = json.loads(line)
        """
        profile = instance["profile"]
        len_profile = len(profile)
        if len_profile >= 10:
            user_profile_feature = profile[0:10]
        else:
            profile.extend([0]*(10-len_profile))
            user_profile_feature = profile
        
        if len(profile) > 1 or (len(profile) == 1 and profile[0] != 0):
            for p in profile:
                if p >= 1 and p <= 65:
                    user_profile_feature[p - 1] = 1
        """
        context_feature = []
        context_feature_fm = []
        dense_feature = [0] * self.dense_length
        plan = instance["plan"]
        for i, val in enumerate(self.dense_feature_list):
            dense_feature[i] = plan[val]

        if (instance["pid"] == ""):
            instance["pid"] = 0

        query = instance["query"]
        weather_dic = instance["weather"]
        for fea in self.pid_list:
            context_feature.append([hash(fea + str(instance[fea])) % self.hash_dim])
            context_feature_fm.append(hash(fea + str(instance[fea])) % self.hash_dim)
        for fea in self.query_feature_list:
            context_feature.append([hash(fea + str(query[fea])) % self.hash_dim])
            context_feature_fm.append(hash(fea + str(query[fea])) % self.hash_dim)
        for fea in self.plan_feature_list:
            context_feature.append([hash(fea + str(plan[fea])) % self.hash_dim])
            context_feature_fm.append(hash(fea + str(plan[fea])) % self.hash_dim)
        for fea in self.rank_feature_list:
            context_feature.append([hash(fea + str(instance[fea])) % self.hash_dim])
            context_feature_fm.append(hash(fea + str(instance[fea])) % self.hash_dim)
        for fea in self.rank_whole_pic_list:
            context_feature.append([hash(fea + str(instance[fea])) % self.hash_dim])
            context_feature_fm.append(hash(fea + str(instance[fea])) % self.hash_dim)
        for fea in self.weather_feature_list:
            context_feature.append([hash(fea + str(weather_dic[fea])) % self.hash_dim])
            context_feature_fm.append(hash(fea + str(weather_dic[fea])) % self.hash_dim)

        label = [int(instance["label"])]

        return dense_feature, context_feature, context_feature_fm, label

    def infer_reader(self, filelist, batch, buf_size):
        print(filelist)

        def local_iter():
            for fname in filelist:
                with open(fname.strip(), "r") as fin:
                    for line in fin:
                        dense_feature, sparse_feature, sparse_feature_fm, label = self._process_line(line)
                        yield [dense_feature] + sparse_feature + [sparse_feature_fm] + [label]

        import paddle
        batch_iter = paddle.batch(
            paddle.reader.shuffle(
                local_iter, buf_size=buf_size),
            batch_size=batch)
        return batch_iter

    #generat inputs for testing
    def test_reader(self, filelist, batch, buf_size):
        print(filelist)

        def local_iter():
            for fname in filelist:
                with open(fname.strip(), "r") as fin:
                    for line in fin:
                        dense_feature, sparse_feature, sparse_feature_fm, label = self._process_line(line)
                        yield [dense_feature] + sparse_feature + [sparse_feature_fm] + [label]

        import paddle
        batch_iter = paddle.batch(
            paddle.reader.buffered(
                local_iter, size=buf_size),
            batch_size=batch)
        return batch_iter

    #generate inputs for trainig 
    def generate_sample(self, line):
        def data_iter():
            dense_feature, sparse_feature, sparse_feature_fm, label = self._process_line(line)
            #feature_name = ["user_profile"]
            feature_name = []
            feature_name.append("dense_feature")
            for idx in self.categorical_range_:
                feature_name.append("context" + str(idx))
            feature_name.append("context_fm")
            feature_name.append("label")
            yield zip(feature_name, [dense_feature] + sparse_feature + [sparse_feature_fm] + [label])

        return data_iter


if __name__ == "__main__":
    map_dataset = MapDataset()
    map_dataset.setup(int(sys.argv[1]))
    map_dataset.run_from_stdin()
