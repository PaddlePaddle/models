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


# path to save data
OUTPUT_train=train
OUTPUT_dev=dev

DATA_URL="https://s3.us-east-2.amazonaws.com/mrqa/release/v2"
alias wget="wget -c --no-check-certificate"
# download training datasets
wget $DATA_URL/train/SQuAD.jsonl.gz -O $OUTPUT_train/SQuAD.jsonl.gz
wget $DATA_URL/train/NewsQA.jsonl.gz -O $OUTPUT_train/NewsQA.jsonl.gz
wget $DATA_URL/train/TriviaQA-web.jsonl.gz -O $OUTPUT_train/TriviaQA.jsonl.gz
wget $DATA_URL/train/SearchQA.jsonl.gz -O $OUTPUT_train/SearchQA.jsonl.gz
wget $DATA_URL/train/HotpotQA.jsonl.gz -O $OUTPUT_train/HotpotQA.jsonl.gz
wget $DATA_URL/train/NaturalQuestionsShort.jsonl.gz -O $OUTPUT_train/NaturalQuestions.jsonl.gz

# download the in-domain development data
wget $DATA_URL/dev/SQuAD.jsonl.gz -O $OUTPUT_dev/SQuAD.jsonl.gz
wget $DATA_URL/dev/NewsQA.jsonl.gz -O $OUTPUT_dev/NewsQA.jsonl.gz
wget $DATA_URL/dev/TriviaQA-web.jsonl.gz -O $OUTPUT_dev/TriviaQA.jsonl.gz
wget $DATA_URL/dev/SearchQA.jsonl.gz -O $OUTPUT_dev/SearchQA.jsonl.gz
wget $DATA_URL/dev/HotpotQA.jsonl.gz -O $OUTPUT_dev/HotpotQA.jsonl.gz
wget $DATA_URL/dev/NaturalQuestionsShort.jsonl.gz -O $OUTPUT_dev/NaturalQuestions.jsonl.gz

# download the out-of-domain development data
wget http://participants-area.bioasq.org/MRQA2019/ -O $OUTPUT_dev/BioASQ.jsonl.gz
wget $DATA_URL/dev/TextbookQA.jsonl.gz -O $OUTPUT_dev/TextbookQA.jsonl.gz
wget $DATA_URL/dev/RelationExtraction.jsonl.gz -O $OUTPUT_dev/RelationExtraction.jsonl.gz
wget $DATA_URL/dev/DROP.jsonl.gz -O $OUTPUT_dev/DROP.jsonl.gz
wget $DATA_URL/dev/DuoRC.ParaphraseRC.jsonl.gz -O $OUTPUT_dev/DuoRC.jsonl.gz
wget $DATA_URL/dev/RACE.jsonl.gz -O $OUTPUT_dev/RACE.jsonl.gz

# check md5sum for training datasets
cd $OUTPUT_train
if md5sum --status -c md5sum_train.txt; then
    echo  "finish download training data"
else
    echo  "md5sum check failed!"
fi
cd ..

# check md5sum for development data
cd $OUTPUT_dev
if md5sum --status -c md5sum_dev.txt; then
    echo  "finish download development data"
else
    echo  "md5sum check failed!"
fi
cd ..

# gzip training datasets
echo "unzipping train data"
NAME_LIST_train="SQuAD NewsQA TriviaQA SearchQA HotpotQA NaturalQuestions"
for name in $NAME_LIST_train;do
    gzip -d $OUTPUT_train/$name.jsonl.gz
done

# gzip development data
echo "unzipping dev data"
NAME_LIST_dev="SQuAD NewsQA TriviaQA SearchQA HotpotQA NaturalQuestions BioASQ TextbookQA RelationExtraction DROP DuoRC RACE"
for name in $NAME_LIST_dev;do
    gzip -d $OUTPUT_dev/$name.jsonl.gz
done
