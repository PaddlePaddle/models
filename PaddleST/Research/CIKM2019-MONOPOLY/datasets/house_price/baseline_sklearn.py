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
File: baseline_sklearn.py
"""

import sys
import numpy as np

from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class CityInfo(object):
    """
        city info
    """
    def __init__(self, city):
        self._set_env(city)
    
    def _set_env(self, city):

        if city == 'sh':
            #sh
            self.business_num = 389
            self.wuye_num = 2931
            self.kfs_num = 4056
            self.age_num = 104
            self.lou_num = 11
            self.average_price = 5.5669712771458115
            self.house_num = 11604
            self.public_num = 970566 + 1
        elif city == 'gz':
            #gz
            self.business_num = 246
            self.wuye_num = 1436
            self.kfs_num = 1753
            self.age_num = 48
            self.lou_num = 12
            self.average_price = 3.120921450522434
            self.house_num = 6508
            self.public_num = 810465 + 1
        elif city == 'sz':
            #sz
            self.business_num = 127
            self.wuye_num = 1096
            self.kfs_num = 1426
            self.age_num = 40
            self.lou_num = 15
            self.average_price = 5.947788464536243
            self.house_num = 3849
            self.public_num = 724320 + 1
        else:#bj, default
            self.business_num = 429
            self.wuye_num = 1548
            self.kfs_num = 1730
            self.age_num = 80
            self.lou_num = 15
            self.average_price = 6.612481698138123
            self.house_num = 7573
            self.public_num = 843426 + 1


if __name__ == '__main__':
    svd = sys.argv[1]
    model = sys.argv[2]

    if model == 'lr':
        clf = linear_model.LinearRegression()
    elif model == 'gb':
        clf = GradientBoostingRegressor()
        #clf = RandomForestRegressor()
        #clf = DecisionTreeRegressor()
    else:
        clf = MLPRegressor(hidden_layer_sizes=(20, ))

    x_train = []
    y_train = []

    x_test = []
    y_test = []
    with open(svd, 'r') as f:
        for line in f:
            ll = line.strip('\r\n').split()
            if ll[0] == 'train':
                y_train.append(float(ll[1]))
                x_train.append(map(float, ll[2:]))
            else:
                y_test.append(float(ll[1]))
                x_test.append(map(float, ll[2:]))

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("%s\t%s\t%s\t%s" % (model, mae, rmse, r2))
 
    
