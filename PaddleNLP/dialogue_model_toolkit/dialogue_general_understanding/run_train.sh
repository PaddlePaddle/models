#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export CPU_NUM=1

TASK_NAME=$1
typeset -l TASK_NAME

BERT_BASE_PATH="./uncased_L-12_H-768_A-12"
INPUT_PATH="./data/${TASK_NAME}"
OUTPUT_PATH="./output/${TASK_NAME}"
PYTHON_PATH="python"

DO_TRAIN=true
DO_VAL=true
DO_TEST=true

#parameter configuration
if [ "${TASK_NAME}" = "udc" ]
then
  save_steps=1000
  max_seq_len=210
  skip_steps=1000
  batch_size=6720
  epoch=2
  learning_rate=2e-5
  DO_VAL=false
  DO_TEST=false
elif [ "${TASK_NAME}" = "swda" ]
then
  save_steps=500
  max_seq_len=128
  skip_steps=200
  batch_size=6720
  epoch=10
  learning_rate=2e-5
elif [ "${TASK_NAME}" = "mrda" ]
then
  save_steps=500
  max_seq_len=128
  skip_steps=200
  batch_size=4096
  epoch=4
  learning_rate=2e-5
elif [ "${TASK_NAME}" = "atis_intent" ]
then
  save_steps=100
  max_seq_len=128
  skip_steps=10
  batch_size=4096
  epoch=20
  learning_rate=2e-5
  INPUT_PATH="./data/atis/${TASK_NAME}"
elif [ "${TASK_NAME}" = "atis_slot" ]
then
  save_steps=100
  max_seq_len=128
  skip_steps=10
  batch_size=32
  epoch=50
  learning_rate=2e-5
  INPUT_PATH="./data/atis/${TASK_NAME}"
elif [ "${TASK_NAME}" = "dstc2" ]
then
  save_steps=400
  max_seq_len=256
  skip_steps=20
  batch_size=8192
  epoch=40
  learning_rate=5e-5
  INPUT_PATH="./data/dstc2/${TASK_NAME}"
else
  echo "not support ${TASK_NAME} dataset.."
  exit 255
fi

# build train, dev, test dataset
cd scripts && sh run_build_data.sh ${TASK_NAME} && cd ..

#training
$PYTHON_PATH -u train.py --task_name ${TASK_NAME} \
                   --use_cuda true\
                   --do_train ${DO_TRAIN} \
                   --do_val ${DO_VAL} \
                   --do_test ${DO_TEST} \
                   --epoch ${epoch} \
                   --batch_size ${batch_size} \
                   --data_dir ${INPUT_PATH} \
                   --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
                   --vocab_path ${BERT_BASE_PATH}/vocab.txt \
                   --init_pretraining_params ${BERT_BASE_PATH}/params \
                   --checkpoints ${OUTPUT_PATH} \
                   --save_steps ${save_steps} \
                   --learning_rate ${learning_rate} \
                   --weight_decay  0.01 \
                   --max_seq_len ${max_seq_len} \
                   --skip_steps ${skip_steps} \
                   --validation_steps 1000000 \
                   --num_iteration_per_drop_scope 10 \
                   --use_fp16 false 

