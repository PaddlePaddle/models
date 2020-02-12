export CUDA_VISIBLE_DEVICES=0
python3 train.py  --batch_size=500     --total_images=1281167    --class_dim=1000 --image_shape=3,224,224 --model_save_dir=output.v2.sing/ --lr_strategy=cosine_decay --lr=0.1  --num_epochs=240  --data_dir=./data/ILSVRC2012 --l2_decay=4e-5  --model=MobileNetV2   --checkpoint=./output.v2.sing/_mobilenet_v2_epoch50 
