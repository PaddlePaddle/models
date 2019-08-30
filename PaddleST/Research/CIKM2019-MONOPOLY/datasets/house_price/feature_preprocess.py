#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
File: feature_preprocess.py
"""
import sys
import random
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import coo_matrix

from datasets.house_price.baseline_sklearn import CityInfo


def parse_line(line, labels, data, row, col, radius, city_info, num_row,
        max_house_num, max_public_num):
    """
        parse line
    """
    ll = line.strip('\r\n').split('\t')
    labels.append(ll[0].split()[0])
    business = int(ll[1].split()[0])
    dis_info = ll[3].split()
    if business >= 0 and business < city_info.business_num:
        data.append(1)
        row.append(num_row)
        col.append(business)

    idx = 0
    h_num = 0
    p_num = 0
    for i in ll[2].split():
        if float(dis_info[idx]) > radius:
            idx += 1
            continue
        if ':' in i: 
            if h_num > max_house_num:
                continue
            h_num += 1
            data.append(1)
            row.append(num_row)
            col.append(city_info.business_num + int(i.split(':')[0]) - 1)
        else:
            if p_num > max_public_num:
                break
            p_num += 1
            data.append(1)
            row.append(num_row)
            col.append(city_info.business_num + city_info.house_num + int(i) - 1)
        idx += 1
             

if __name__ == '__main__':
    test = sys.argv[1]
    radius = float(sys.argv[2])
    max_house_num = float(sys.argv[3])
    max_public_num = float(sys.argv[4])
    city_info = CityInfo(sys.argv[5])

    train_data = []
    train_row = []
    train_col = []
    train_labels = []
 
    num_row = 0
    for line in sys.stdin:
        parse_line(line, train_labels, train_data, train_row, train_col, radius, city_info,
                num_row, max_house_num, max_public_num) 
        num_row += 1 

    coo = coo_matrix((train_data, (train_row, train_col)),
            shape=(num_row, city_info.business_num + city_info.house_num + city_info.public_num))
    
    svd = TruncatedSVD(n_components=200, n_iter=10, random_state=0)
    svd.fit(coo.tocsr())

    x_train = svd.transform(coo.tocsr())
    for i in range(len(x_train)):
        print("train %s %s" % (train_labels[i], " ".join(map(str, x_train[i]))))
    
    test_data = []
    test_row = []
    test_col = []
    test_labels = []
      
    with open(test, 'r') as f:
        num_row = 0
        for line in f:
            parse_line(line, test_labels, test_data, test_row, test_col, radius, city_info,
                    num_row, max_house_num, max_public_num) 
            num_row += 1 

    coo = coo_matrix((test_data, (test_row, test_col)),
            shape=(num_row, city_info.business_num + city_info.house_num + city_info.public_num))
    x_test = svd.transform(coo.tocsr())
    for i in range(len(x_test)):
        print("test %s %s" % (test_labels[i], " ".join(map(str, x_test[i]))))
