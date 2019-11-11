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

# prepare output directory
if [ -d $OUTPUT ]; then
    rm -rf $OUTPUT 
fi
mkdir $OUTPUT 

# begin training
echo ">> Begin train now"
python3 -u ./bin/run.py \
 --dataset $TASK \
 --vocab_size $NUM_VOCAB \
 --num_relations $NUM_RELATIONS \
 --use_cuda true \
 --do_train true \
 --do_predict false \
 --train_file $TRAIN_FILE \
 --predict_file $TEST_FILE \
 --max_seq_len $MAX_SEQ_LEN \
 --checkpoints $CHECKPOINTS \
 --soft_label $SOFT_LABEL \
 --batch_size $BATCH_SIZE \
 --epoch $EPOCH \
 --learning_rate $LEARNING_RATE \
 --hidden_dropout_prob $HIDDEN_DROPOUT_PROB \
 --attention_probs_dropout_prob $ATTENTION_PROBS_DROPOUT_PROB \
 --skip_steps $SKIP_STEPS \
 --vocab_path $VOCAB_PATH \
 --hidden_size $HIDDEN_SIZE \
 --sen_candli_file $SEN_CANDLI_PATH \
 --sen_trivial_file $TRIVAL_SEN_PATH \
 --num_hidden_layers $NUM_HIDDEN_LAYERS \
 --num_attention_heads $NUM_ATTENTION_HEADS \
 --max_position_embeddings $MAX_POSITION_EMBEDDINS \
 --use_ema false > $LOG_FILE 2>&1

