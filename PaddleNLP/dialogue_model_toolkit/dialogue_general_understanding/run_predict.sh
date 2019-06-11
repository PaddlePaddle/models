#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
export CPU_NUM=1

TASK_NAME=$1
BERT_BASE_PATH="./uncased_L-12_H-768_A-12"
INPUT_PATH="./data/${TASK_NAME}"
OUTPUT_PATH="./output/${TASK_NAME}"
PYTHON_PATH="python"

if [ "$TASK_NAME" = "udc" ]
then
  best_model="step_62500"
  max_seq_len=210
  batch_size=6720
elif [ "$TASK_NAME" = "swda" ] 
then
  best_model="step_12500"
  max_seq_len=128
  batch_size=6720
elif [ "$TASK_NAME" = "mrda" ]
then
  best_model="step_6500"
  max_seq_len=128
  batch_size=6720
elif [ "$TASK_NAME" = "atis_intent" ]
then
  best_model="step_600"
  max_seq_len=128
  batch_size=4096
  INPUT_PATH="./data/atis/${TASK_NAME}"
elif [ "$TASK_NAME" = "atis_slot" ]
then
  best_model="step_7500"
  max_seq_len=128
  batch_size=32
  INPUT_PATH="./data/atis/${TASK_NAME}"
elif [ "$TASK_NAME" = "dstc2" ]
then
  best_model="step_12000"
  max_seq_len=700
  batch_size=6000
  INPUT_PATH="./data/dstc2/${TASK_NAME}"
else
  echo "not support ${TASK_NAME} dataset.."   
  exit 255
fi

$PYTHON_PATH -u predict.py --task_name ${TASK_NAME} \
                   --use_cuda true\
                   --batch_size ${batch_size} \
                   --init_checkpoint ${OUTPUT_PATH}/${best_model} \
                   --data_dir ${INPUT_PATH} \
                   --vocab_path ${BERT_BASE_PATH}/vocab.txt \
                   --max_seq_len ${max_seq_len} \
                   --bert_config_path ${BERT_BASE_PATH}/bert_config.json

