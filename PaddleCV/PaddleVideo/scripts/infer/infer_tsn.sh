python infer.py --model_name="TSN" --config=./configs/tsn.txt --filelist=./dataset/kinetics/infer.list \
                --log_interval=10 --weights=./checkpoints/TSN_epoch0 --save_dir=./save
