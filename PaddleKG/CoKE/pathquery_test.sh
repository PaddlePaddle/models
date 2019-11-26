#! /bin/bash
set -exu
set -o pipefail

#configs
CONF_FP=$1
CUDA_ID=$2


source $CONF_FP
export CUDA_VISIBLE_DEVICES=$CUDA_ID
export FLAGS_sync_nccl_allreduce=1

# todo: modify to your own path
export LD_LIBRARY_PATH=$(pwd)/env/lib/nccl2.3.7_cuda9.0/lib:/home/work/cudnn/cudnn_v7/cuda/lib64:/home/work/cuda-9.0/extras/CUPTI/lib64/:/home/work/cuda-9.0/lib64/:$LD_LIBRARY_PATH


max_step_id=`ls $CHECKPOINTS | grep "step" | awk -F"_" '{print $NF}' | grep -v "Found"  |sort -n |tail -1`
INIT_CHECKPOINT_STEP=${CHECKPOINTS}/step_${max_step_id}
echo "max_step_id: $max_step_id"

echo ">> Begin predict now"
python3 -u ./bin/run.py \
 --dataset $TASK \
 --vocab_size $NUM_VOCAB \
 --num_relations $NUM_RELATIONS \
 --use_cuda true \
 --do_train false \
 --do_predict true \
 --predict_file $TEST_FILE \
 --init_checkpoint ${INIT_CHECKPOINT_STEP} \
 --batch_size $BATCH_SIZE \
 --vocab_path $VOCAB_PATH \
 --sen_candli_file $SEN_CANDLI_PATH \
 --sen_trivial_file $TRIVAL_SEN_PATH \
 --max_seq_len $MAX_SEQ_LEN \
 --learning_rate $LEARNING_RATE \
 --use_ema false > $LOG_EVAL_FILE 2>&1

