export CUDA_VISIBLE_DEVICES=1
python3.7 eval.py \
          --config=slowfast-single.yaml \
          --use_gpu=True \
          --use_data_parallel=0 \
          --weights=checkpoints/slowfast_epoch0_00000
