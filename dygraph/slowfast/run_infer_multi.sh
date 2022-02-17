export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3.7 -m paddle.distributed.launch \
          predict.py \
          --config=slowfast.yaml \
          --use_gpu=True \
          --use_data_parallel=1 \
          --weights=checkpoints/slowfast_epoch195
