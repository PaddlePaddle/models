export CUDA_VISIBLE_DEVICES=2
python3.7 predict.py \
          --config=slowfast-single.yaml \
          --use_gpu=True \
          --use_data_parallel=0 \
          --weights=checkpoints/slowfast_epoch00000
