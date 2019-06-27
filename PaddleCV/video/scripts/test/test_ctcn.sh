export CUDA_VISIBLE_DEVICES=0
python test.py --model_name="CTCN" --config=./configs/ctcn.txt \
                --log_interval=1 --weights=./checkpoints/CTCN_epoch0
