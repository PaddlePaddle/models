#!/bin/bash

input_file=$1
output_file=$2

# convert the data from MARCO V2 (json) format to MARCO V1 (jsonl) format. 
# the script was forked from MARCO repo. 
# the format of MARCO V1 is much more easier to explore. 
python3 marcov2_to_v1_tojsonl.py $input_file $input_file.marcov1

# convert the data from MARCO V1 format to DuReader format. 
python3 marcov1_to_dureader.py $input_file.marcov1 >$input_file.dureader_raw

# tokenize the data. 
python3 marco_tokenize_data.py $input_file.dureader_raw >$input_file.segmented

# find fake answers (indicating the start and end positions of answers in the document) for train and dev sets. 
# note that this should not be applied for test set, since there is no ground truth in test set. 
python preprocess.py $input_file.segmented >$output_file

# remove the temporal data files. 
rm -rf $input_file.dureader_raw $input_file.segmented
