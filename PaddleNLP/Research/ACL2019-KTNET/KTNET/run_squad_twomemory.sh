#!/bin/bash
set -x
set -e 

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8

# echo "running run.sh..."

if [ ! -d log ]; then
mkdir log
else
rm -r log/*
fi

if [ ! -d output ]; then
mkdir output
else
rm -r output/*
fi


PWD_DIR=`pwd`

export PATH=$PWD_DIR/python/bin/:./python/lib/:$PATH
export PYTHONPATH=$PWD_DIR/python/lib/:$PWD_DIR/python/lib/python3.7/site-packages/:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD_DIR/nccl2.3.7_cuda9.0/lib:/home/work/cuda-9.0/lib64:/home/work/cudnn/cudnn_v7/cuda/lib64:/home/work/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export FLAGS_cudnn_deterministic=true
export FLAGS_cpu_deterministic=true

echo `which python` > $PWD_DIR/log/info.log

python -c 'import sys; print(sys.path)' >> $PWD_DIR/log/info.log
echo $WORK_DIR >> $PWD_DIR/log/info.log

DATA=$PWD_DIR/data/
#BERT_DIR=$PWD_DIR/uncased_L-12_H-768_A-12
#BERT_DIR=$PWD_DIR/cased_L-12_H-768_A-12
#BERT_DIR=$PWD_DIR/uncased_L-24_H-1024_A-16
BERT_DIR=$PWD_DIR/cased_L-24_H-1024_A-16

WN_CPT_EMBEDDING_PATH=$PWD_DIR/concept_resources/embeddings/wn_concept2vec.txt
NELL_CPT_EMBEDDING_PATH=$PWD_DIR/concept_resources/embeddings/nell_concept2vec.txt

python3 src/run_squad_twomemory.py \
  --batch_size 6 \
  --do_train true \
  --do_predict true \
  --do_lower_case false \
  --init_pretraining_params $BERT_DIR/params \
  --train_file $DATA/SQuAD/train-v1.1.json \
  --predict_file $DATA/SQuAD/dev-v1.1.json \
  --vocab_path $BERT_DIR/vocab.txt \
  --bert_config_path $BERT_DIR/bert_config.json \
  --freeze false \
  --save_steps 4000 \
  --validation_steps 4000 \
  --weight_decay 0.01 \
  --warmup_proportion 0.1 \
  --learning_rate 3e-5 \
  --epoch 2 \
  --max_seq_len 384 \
  --doc_stride 128 \
  --wn_concept_embedding_path $WN_CPT_EMBEDDING_PATH \
  --nell_concept_embedding_path $NELL_CPT_EMBEDDING_PATH \
  --use_wordnet true \
  --use_nell true \
  --random_seed 44 \
  --checkpoints output/ 1>$PWD_DIR/log/train.log 2>&1

cd output
find . -mindepth 1 -maxdepth 1 -type d -exec sh -c 'tar czvf $(basename {}).tar.gz $(basename {})' \;
find . -mindepth 1 -maxdepth 1 -type d -exec sh -c 'rm -rf {}' \;
cd ..