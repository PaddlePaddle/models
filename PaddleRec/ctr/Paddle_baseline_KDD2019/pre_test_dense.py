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


import os, sys, time, random, csv, datetime, json
import pandas as pd
import numpy as np
import argparse
import logging
import time

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("preprocess")
logger.setLevel(logging.INFO)

TRAIN_QUERIES_PATH = "./data_set_phase1/test_queries.csv"
TRAIN_PLANS_PATH = "./data_set_phase1/test_plans.csv"
TRAIN_CLICK_PATH = "./data_set_phase1/train_clicks.csv"
PROFILES_PATH = "./data_set_phase1/profiles.csv"

O1_MIN = 115.47
O1_MAX = 117.29

O2_MIN = 39.46
O2_MAX = 40.97

D1_MIN = 115.44
D1_MAX = 117.37

D2_MIN = 39.46
D2_MAX = 40.96

DISTANCE_MIN = 1.0
DISTANCE_MAX = 225864.0
THRESHOLD_DIS = 200000.0

PRICE_MIN = 200.0
PRICE_MAX = 92300.0
THRESHOLD_PRICE = 20000

ETA_MIN = 1.0
ETA_MAX = 72992.0
THRESHOLD_ETA = 10800.0


def build_norm_feature():
    with open("./out/normed_test_session.txt", 'w') as nf:
        with open("./out/test_session.txt", 'r') as f:
            for line in f:
                cur_map = json.loads(line)

                cur_map["plan"]["distance"] = (cur_map["plan"]["distance"] - DISTANCE_MIN) / (DISTANCE_MAX - DISTANCE_MIN)

                if cur_map["plan"]["price"]:
                    cur_map["plan"]["price"] = (cur_map["plan"]["price"] - PRICE_MIN) / (PRICE_MAX - PRICE_MIN)
                else:
                    cur_map["plan"]["price"] = 0.0

                cur_map["plan"]["eta"] = (cur_map["plan"]["eta"] - ETA_MIN) / (ETA_MAX - ETA_MIN)

                cur_json_instance = json.dumps(cur_map)
                nf.write(cur_json_instance + '\n')


def preprocess():
    """
    Construct the train data indexed by session id and mode id jointly. Convert all the raw features (user profile,
    od pair, req time, click time, eta, price, distance, transport mode) to one-hot ids used for
    embedding. We split the one-hot features into two categories: user feature and context feature for
    better understanding of FFM algorithm.
    Note that the user profile is already provided by one-hot encoded form, we convert it back to the
    ids for unity with the context feature and easily using of PaddlePaddle embedding layer. Given the
    train clicks data, we label each train instance with 1 or 0 depend on if this instance is clicked or
    not.
    :return:
    """
    #args = parse_args()

    train_data_dict = {}
    with open("./weather.json", 'r') as f:
        weather_dict = json.load(f)

    with open(TRAIN_QUERIES_PATH, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        train_index_list = []
        for k, line in enumerate(csv_reader):
            if k == 0: continue
            if line[0] == "": continue
            if line[1] == "":
                train_index_list.append(line[0] + "_0")
            else:
                train_index_list.append(line[0] + "_" + line[1])

            train_index = line[0]
            train_data_dict[train_index] = {}
            train_data_dict[train_index]["pid"] = line[1]
            train_data_dict[train_index]["query"] = {}

            reqweekday = datetime.datetime.strptime(line[2], '%Y-%m-%d %H:%M:%S').strftime("%w")
            reqhour = datetime.datetime.strptime(line[2], '%Y-%m-%d %H:%M:%S').strftime("%H")

            date_key = datetime.datetime.strptime(line[2], '%Y-%m-%d %H:%M:%S').strftime("%m-%d")
            train_data_dict[train_index]["weather"] = {}
            train_data_dict[train_index]["weather"].update({"max_temp": weather_dict[date_key]["max_temp"]})
            train_data_dict[train_index]["weather"].update({"min_temp": weather_dict[date_key]["min_temp"]})
            train_data_dict[train_index]["weather"].update({"wea": weather_dict[date_key]["weather"]})
            train_data_dict[train_index]["weather"].update({"wind": weather_dict[date_key]["wind"]})

            train_data_dict[train_index]["query"].update({"weekday":reqweekday})
            train_data_dict[train_index]["query"].update({"hour":reqhour})

            o = line[3].split(',')
            o_first = o[0]
            o_second = o[1]
            train_data_dict[train_index]["query"].update({"o1":float(o_first)})
            train_data_dict[train_index]["query"].update({"o2":float(o_second)})

            d = line[4].split(',')
            d_first = d[0]
            d_second = d[1]
            train_data_dict[train_index]["query"].update({"d1":float(d_first)})
            train_data_dict[train_index]["query"].update({"d2":float(d_second)})

    plan_map = {}
    plan_data = pd.read_csv(TRAIN_PLANS_PATH)
    for index, row in plan_data.iterrows():
        plans_str = row['plans']
        plans_list = json.loads(plans_str)
        session_id = str(row['sid'])
        # train_data_dict[session_id]["plans"] = []
        plan_map[session_id] = plans_list

    profile_map = {}
    with open(PROFILES_PATH, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for k, line in enumerate(csv_reader):
            if k == 0: continue
            profile_map[line[0]] = [i for i in range(len(line)) if line[i] == "1.0"]

    session_click_map = {}
    with open(TRAIN_CLICK_PATH, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for k, line in enumerate(csv_reader):
            if k == 0: continue
            if line[0] == "" or line[1] == "" or line[2] == "":
                continue
            session_click_map[line[0]] = line[2]
    #return train_data_dict, profile_map, session_click_map, plan_map
    generate_sparse_features(train_data_dict, profile_map, session_click_map, plan_map)


def generate_sparse_features(train_data_dict, profile_map, session_click_map, plan_map):
    if not os.path.isdir("./out/"):
        os.mkdir("./out/")
    with open(os.path.join("./out/", "test_session.txt"), 'w') as f_train:
        for session_id, plan_list in plan_map.items():
            if session_id not in train_data_dict:
                continue
            cur_map = train_data_dict[session_id]
            cur_map["session_id"] = session_id
            if cur_map["pid"] != "":
                cur_map["profile"] = profile_map[cur_map["pid"]]
            else:
                cur_map["profile"] = [0]
            # del cur_map["pid"]
            whole_rank = 0
            for plan in plan_list:
                whole_rank += 1
                cur_map["mode_rank" + str(whole_rank)] = plan["transport_mode"]

            if whole_rank < 5:
                for r in range(whole_rank + 1, 6):
                    cur_map["mode_rank" + str(r)] = -1

            cur_map["whole_rank"] = whole_rank
            rank = 1

            price_list = []
            eta_list = []
            distance_list = []
            for plan in plan_list:
                if not plan["price"]:
                    price_list.append(0)
                else:
                    price_list.append(int(plan["price"]))
                eta_list.append(int(plan["eta"]))
                distance_list.append(int(plan["distance"]))
            price_list.sort(reverse=False)
            eta_list.sort(reverse=False)
            distance_list.sort(reverse=False)

            for plan in plan_list:
                if plan["price"] and int(plan["price"]) == price_list[0]:
                    cur_map["mode_min_price"] = plan["transport_mode"]
                if plan["price"] and int(plan["price"]) == price_list[-1]:
                    cur_map["mode_max_price"] = plan["transport_mode"]
                if int(plan["eta"]) == eta_list[0]:
                    cur_map["mode_min_eta"] = plan["transport_mode"]
                if int(plan["eta"]) == eta_list[-1]:
                    cur_map["mode_max_eta"] = plan["transport_mode"]
                if int(plan["distance"]) == distance_list[0]:
                    cur_map["mode_min_distance"] = plan["transport_mode"]
                if int(plan["distance"]) == distance_list[-1]:
                    cur_map["mode_max_distance"] = plan["transport_mode"]
            if "mode_min_price" not in cur_map:
                cur_map["mode_min_price"] = -1
            if "mode_max_price" not in cur_map:
                cur_map["mode_max_price"] = -1

            for plan in plan_list:
                cur_price = int(plan["price"]) if plan["price"] else 0
                cur_eta = int(plan["eta"])
                cur_distance = int(plan["distance"])
                cur_map["price_rank"] = price_list.index(cur_price) + 1
                cur_map["eta_rank"] = eta_list.index(cur_eta) + 1
                cur_map["distance_rank"] = distance_list.index(cur_distance) + 1

                if ("transport_mode" in plan) and (session_id in session_click_map) and (
                        int(plan["transport_mode"]) == int(session_click_map[session_id])):
                    cur_map["plan"] = plan
                    cur_map["label"] = 1
                else:
                    cur_map["plan"] = plan
                    cur_map["label"] = 0

                cur_map["plan_rank"] = rank
                rank += 1
                cur_json_instance = json.dumps(cur_map)
                f_train.write(cur_json_instance + '\n')

            cur_map["plan"]["distance"] = -1
            cur_map["plan"]["price"] = -1
            cur_map["plan"]["eta"] = -1
            cur_map["plan"]["transport_mode"] = 0
            cur_map["plan_rank"] = 0
            cur_map["price_rank"] = 0
            cur_map["eta_rank"] = 0
            cur_map["plan_rank"] = 0
            cur_map["label"] = 1
            cur_json_instance = json.dumps(cur_map)
            f_train.write(cur_json_instance + '\n')


    build_norm_feature()


if __name__ == "__main__":
    preprocess()