export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py --model_name="STNET" --config=./configs/stnet.txt --epoch=60 \
                --valid_interval=1 --log_interval=10
