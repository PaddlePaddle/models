export CUDA_VISIBLE_DEVICES=0
python3 train.py      --batch_size=128        --total_images=1281167    --class_dim=1000 --image_shape=3,224,224 --model_save_dir=output/ --lr_strategy=piecewise_decay --lr=0.1   --data_dir=../../PaddleCV/image_classification/data/ILSVRC2012  --model=MobileNetV2
