set -eu 
set -o pipefail 


# Attention! Python 2.7.14  and python3 gives different vocabulary order. We use Python 2.7.14 to preprocess files.

# input files: train.txt valid.txt test.txt  
# (these are default filenames, change files name with the following arguments:  --train $trainname --valid $validname --test $testname)
# output files: vocab.txt train.coke.txt valid.coke.txt test.coke.txt
python ./bin/kbc_data_preprocess.py --task fb15k --dir ./data/fb15k
python ./bin/kbc_data_preprocess.py --task wn18 --dir ./data/wn18
python ./bin/kbc_data_preprocess.py --task fb15k237 --dir ./data/fb15k237
python ./bin/kbc_data_preprocess.py --task wn18rr --dir ./data/wn18rr

# input files: train dev test
# (these are default filenames, change files name with the following arguments: --train $trainname --valid $validname --test $testname)
# output files: vocab.txt train.coke.txt valid.coke.txt test.coke.txt sen_candli.txt trivial_sen.txt
python ./bin/pathquery_data_preprocess.py --task pathqueryFB --dir ./data/pathqueryFB 
python ./bin/pathquery_data_preprocess.py --task pathqueryWN --dir ./data/pathqueryWN
