#! /bin/sh

tmp_dir=wmt16_en_de
data_dir=data
source_file=train.tok.clean.bpe.32000.en
target_file=train.tok.clean.bpe.32000.de
source_vocab_size=32000
target_vocab_size=32000
num_shards=100

if [ ! -d wmt16_en_de ]
then
    mkdir wmt16_en_de
fi

wget https://baidu-nlp.bj.bcebos.com/EMNLP2019-MAL/wmt16_en_de.tar.gz -O wmt16_en_de/wmt16_en_de.tar.gz
tar -zxf wmt16_en_de/wmt16_en_de.tar.gz -C wmt16_en_de

if [ ! -d $data_dir ]
then
    mkdir data
fi

if [ ! -d testset ]
then
    mkdir testset
fi


cp wmt16_en_de/vocab.bpe.32000 data/vocab.source.32000

python ./src/gen_records.py --tmp_dir ${tmp_dir} --data_dir ${data_dir} --source_train_files ${source_file} --target_train_files ${target_file} --source_vocab_size ${source_vocab_size} --target_vocab_size ${target_vocab_size} --num_shards ${num_shards} --token True --onevocab True

python ./src/preprocess/gen_utils.py --vocab $data_dir/vocab.source.${source_vocab_size} --testset ${tmp_dir}/newstest2014.tok.bpe.32000.en --output ./testset/testfile
