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
PATH_dev=./data/input/mrqa_evaluation_dataset


# path of dev predict
KD_prediction=./prediction_results/KD_ema_predictions.json

files=$(ls ./prediction_results/*.log 2> /dev/null | wc -l)
if [ "$files" != "0" ];
then
    rm prediction_results/*.log
fi

# evaluation KD model
echo "evaluate knowledge distillation model........................................."
for dataset in `ls $PATH_dev/in_domain_dev/*.raw.json`;do
    echo $dataset >> prediction_results/KD.log
    python ../multi_task_learning/scripts/evaluate-v1.1.py $dataset $KD_prediction >> prediction_results/KD.log
done

for dataset in `ls $PATH_dev/out_of_domain_dev/*.raw.json`;do
    echo $dataset >> prediction_results/KD.log
    python ../multi_task_learning/scripts/evaluate-v1.1.py $dataset $KD_prediction >> prediction_results/KD.log
done
python ../multi_task_learning/scripts/macro_avg.py prediction_results/KD.log








