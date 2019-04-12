#!/bin/bash
set -e

enable_dgc=False

while true ; do
  case "$1" in
    -enable_dgc) enable_dgc="$2" ; shift 2 ;;
    *)
       if [[ ${#1} > 0 ]]; then
          echo "not supported arugments ${1}" ; exit 1 ;
       else
           break
       fi
       ;;
  esac
done

case "${enable_dgc}" in
    True) ;;
    False) ;;
    *) echo "not support argument -enable_dgc: ${dgc}" ; exit 1 ;;
esac

export MODEL="DistResNet"
export PADDLE_TRAINER_ENDPOINTS="127.0.0.1:7160,127.0.0.1:7161"
# PADDLE_TRAINERS_NUM is used only for reader when nccl2 mode
export PADDLE_TRAINERS_NUM="2"

mkdir -p logs

# NOTE: set NCCL_P2P_DISABLE so that can run nccl2 distribute train on one node.

# You can set vlog to see more details' log.
# export GLOG_v=1
# export GLOG_logtostderr=1

PADDLE_TRAINING_ROLE="TRAINER" \
PADDLE_CURRENT_ENDPOINT="127.0.0.1:7160" \
PADDLE_TRAINER_ID="0" \
CUDA_VISIBLE_DEVICES="0" \
NCCL_P2P_DISABLE="1" \
python -u dist_train.py --enable_dgc ${enable_dgc} --model $MODEL --update_method nccl2 --batch_size 32   &> logs/tr0.log &

PADDLE_TRAINING_ROLE="TRAINER" \
PADDLE_CURRENT_ENDPOINT="127.0.0.1:7161" \
PADDLE_TRAINER_ID="1" \
CUDA_VISIBLE_DEVICES="1" \
NCCL_P2P_DISABLE="1" \
python -u dist_train.py --enable_dgc ${enable_dgc} --model $MODEL --update_method nccl2 --batch_size 32  &> logs/tr1.log &
