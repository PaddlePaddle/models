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
PATH_dev=./PALM/data/mrqa_dev
# path of dev prediction
BERT_MLM_PATH_prediction=./prediction_results/BERT_MLM_ema_predictions.json 
BERT_MLM_ParaRank_PATH_prediction=./prediction_results/BERT_MLM_ParaRank_ema_predictions.json

files=$(ls ./prediction_results/*.log 2> /dev/null | wc -l)
if [ "$files" != "0" ];
then
    rm prediction_results/BERT_MLM*.log
fi

# evaluation BERT_MLM
echo "evaluate BERT_MLM model........................................."
for dataset in `ls $PATH_dev/in_domain_dev/*.raw.json`;do
    echo $dataset >> prediction_results/BERT_MLM.log
    python scripts/evaluate-v1.1.py $dataset $BERT_MLM_PATH_prediction >> prediction_results/BERT_MLM.log
done

for dataset in `ls $PATH_dev/out_of_domain_dev/*.raw.json`;do
    echo $dataset >> prediction_results/BERT_MLM.log
    python scripts/evaluate-v1.1.py $dataset $BERT_MLM_PATH_prediction >> prediction_results/BERT_MLM.log
done
python scripts/macro_avg.py prediction_results/BERT_MLM.log

# evaluation BERT_MLM_ParaRank_PATH_prediction
echo "evaluate BERT_MLM_ParaRank model................................"
for dataset in `ls $PATH_dev/in_domain_dev/*.raw.json`;do
    echo $dataset >> prediction_results/BERT_MLM_ParaRank.log
    python scripts/evaluate-v1.1.py $dataset $BERT_MLM_ParaRank_PATH_prediction >> prediction_results/BERT_MLM_ParaRank.log
done


for dataset in `ls $PATH_dev/out_of_domain_dev/*.raw.json`;do
    echo $dataset >> prediction_results/BERT_MLM_ParaRank.log
    python scripts/evaluate-v1.1.py $dataset $BERT_MLM_ParaRank_PATH_prediction >> prediction_results/BERT_MLM_ParaRank.log
done
python scripts/macro_avg.py prediction_results/BERT_MLM_ParaRank.log







