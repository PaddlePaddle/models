export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch --log_dir ./mylog.v2 train.py --use_data_parallel 1 --batch_size=500     --total_images=1281167    --class_dim=1000 --image_shape=3,224,224 --model_save_dir=output.v2.mul/ --lr_strategy=cosine_decay --lr=0.1  --num_epochs=240  --data_dir=./data/ILSVRC2012 --l2_decay=4e-5  --model=MobileNetV2
