#!/usr/bin/env bash
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# path of dev data
PATH_dev=./data/dev
# path of dev prediction
PATH_prediction=./output/ema_predictions.json

# evaluation
for dataset in `ls $PATH_dev/*.raw.json`;do
    if [ "$dataset" = "./data/dev/mrqa-combined.raw.json" ]; then
        continue
    fi
    echo $dataset
    python evaluate-v1.1.py $dataset $PATH_prediction
done
