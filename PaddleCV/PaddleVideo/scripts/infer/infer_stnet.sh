python infer.py --model_name="STNET" --config=./configs/stnet.txt --filelist=./dataset/kinetics/infer.list \
                --log_interval=10 --weights=./checkpoints/STNET_epoch0 --save_dir=./save
