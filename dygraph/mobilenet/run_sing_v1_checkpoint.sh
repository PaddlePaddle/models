export CUDA_VISIBLE_DEVICES=0
python3 train.py      --batch_size=256        --total_images=1281167    --class_dim=1000 --image_shape=3,224,224 --model_save_dir=output.v1.sing/ --lr_strategy=piecewise_decay --lr=0.1   --data_dir=./data/ILSVRC2012  --l2_decay=3e-5  --model=MobileNetV1   --checkpoint=./output.v1.sing/_mobilenet_v1_epoch50 
