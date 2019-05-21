export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py --model_name="NEXTVLAD" --config=./configs/nextvlad.txt --epoch=6 \
                --valid_interval=1 --log_interval=10
