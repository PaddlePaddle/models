#! /bin/bash

#==========
set -e
set -x
set -u
set -o pipefail
#==========

#==========configs
conf_fp=$1
CUDA=$2
source $conf_fp

#=========init env
export CUDA_VISIBLE_DEVICES=$CUDA
export FLAGS_sync_nccl_allreduce=1

#modify to your own path
export LD_LIBRARY_PATH=$(pwd)/env/lib/nccl2.3.7_cuda9.0/lib:/home/work/cudnn/cudnn_v7/cuda/lib64:/home/work/cuda-9.0/extras/CUPTI/lib64/:/home/work/cuda-9.0/lib64/:$LD_LIBRARY_PATH

#======beging train
if [ -d $OUTPUT ]; then
    rm -rf $OUTPUT 
fi
mkdir $OUTPUT 


max_step_id=`ls $INIT_CHECKPOINTS | grep "step" | awk -F"_" '{print $NF}' | grep -v "Found"  |sort -n |tail -1`
INIT_CHECKPOINT_STEP=${INIT_CHECKPOINTS}/step_${max_step_id}
echo "init_checkpoints_steps: $max_step_id"


#--init_checkpoint ${INIT_CHECKPOINT}
echo ">> Begin kbc test now, log file: $LOG_EVAL_FILE"
python3 -u ./bin/run.py \
 --dataset $TASK \
 --vocab_size $NUM_VOCAB \
 --num_relations $NUM_RELATIONS \
 --use_cuda true \
 --do_train false \
 --train_file $TRAIN_FILE \
 --checkpoints $CHECKPOINTS \
 --init_checkpoint ${INIT_CHECKPOINT_STEP} \
 --true_triple_path $TRUE_TRIPLE_PATH \
 --max_seq_len $MAX_SEQ_LEN \
 --soft_label $SOFT_LABEL \
 --batch_size $BATCH_SIZE \
 --epoch $EPOCH \
 --learning_rate $LEARNING_RATE \
 --hidden_dropout_prob $HIDDEN_DROPOUT_PROB \
 --attention_probs_dropout_prob $ATTENTION_PROBS_DROPOUT_PROB \
 --skip_steps $SKIP_STEPS \
 --do_predict true \
 --predict_file $TEST_FILE \
 --vocab_path $VOCAB_PATH \
 --hidden_size $HIDDEN_SIZE \
 --num_hidden_layers $NUM_HIDDEN_LAYERS \
 --num_attention_heads $NUM_ATTENTION_HEADS \
 --max_position_embeddings $MAX_POSITION_EMBEDDINS \
 --use_ema false > $LOG_EVAL_FILE 2>&1

echo ">> Finish kbc test, log file: $LOG_EVAL_FILE"
