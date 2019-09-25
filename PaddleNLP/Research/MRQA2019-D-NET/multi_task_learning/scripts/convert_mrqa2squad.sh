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

# path of train and dev data
PATH_train=train
PATH_dev=dev

# Convert train data from MRQA format to SQuAD format
NAME_LIST_train="SQuAD NewsQA TriviaQA SearchQA HotpotQA NaturalQuestions"
for name in $NAME_LIST_train;do
    echo "Converting training data from MRQA format to SQuAD format: ""$name"
    python convert_mrqa2squad.py $PATH_train/$name.jsonl
done

# Convert dev data from MRQA format to SQuAD format
NAME_LIST_dev="SQuAD NewsQA TriviaQA SearchQA HotpotQA NaturalQuestions BioASQ TextbookQA RelationExtraction DROP DuoRC RACE"
for name in $NAME_LIST_dev;do
    echo "Converting development data from MRQA format to SQuAD format: ""$name"
    python convert_mrqa2squad.py --dev $PATH_dev/$name.jsonl
done
