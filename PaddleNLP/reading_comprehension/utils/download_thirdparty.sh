#!/bin/bash
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

# We use Bleu and Rouge as evaluation metrics, the calculation of these metrics
# relies on the scoring scripts under "https://github.com/tylin/coco-caption"

bleu_base_url='https://raw.githubusercontent.com/tylin/coco-caption/master/pycocoevalcap/bleu'
bleu_files=("LICENSE" "__init__.py" "bleu.py" "bleu_scorer.py")

rouge_base_url="https://raw.githubusercontent.com/tylin/coco-caption/master/pycocoevalcap/rouge"
rouge_files=("__init__.py" "rouge.py")

download() {
    local metric=$1; shift;
    local base_url=$1; shift;
    local fnames=($@);

    mkdir -p ${metric}
    for fname in ${fnames[@]};
    do
        printf "downloading: %s\n" ${base_url}/${fname}
        wget --no-check-certificate ${base_url}/${fname} -O ${metric}/${fname}
    done
}

# prepare rouge
download "rouge_metric" ${rouge_base_url} ${rouge_files[@]}

# prepare bleu
download "bleu_metric" ${bleu_base_url} ${bleu_files[@]}

# convert python 2.x source code to python 3.x
2to3 -w "../utils/bleu_metric/bleu_scorer.py"
2to3 -w "../utils/bleu_metric/bleu.py"
