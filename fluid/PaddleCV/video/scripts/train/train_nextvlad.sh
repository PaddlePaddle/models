export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py --model-name="NEXTVLAD" --config=./configs/nextvlad.txt --epoch-num=6 \
                --valid-interval=1 --log-interval=10
