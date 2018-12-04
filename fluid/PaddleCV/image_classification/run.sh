#Hyperparameters config
python train.py \
       --model=SE_ResNeXt50_32x4d \
       --batch_size=32 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --with_mem_opt=False \
       --lr_strategy=piecewise_decay \
       --lr=0.1
#      >log_SE_ResNeXt50_32x4d.txt 2>&1 &

#AlexNet:
#python train.py \
#       --model=AlexNet \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=False \
#       --lr_strategy=piecewise_decay \
#	--num_epochs=120 \
#       --lr=0.01

#VGG11:
#python train.py \
#       --model=VGG11 \
#       --batch_size=512 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=False \
#       --lr_strategy=piecewise_decay \
#       --num_epochs=120 \
#       --lr=0.1


#MobileNet v1:
#python train.py \
#       --model=MobileNet \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=False \
#       --lr_strategy=piecewise_decay \
#	--num_epochs=120 \
#       --lr=0.1


#ResNet50:
#python train.py \
#       --model=ResNet50 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=False \
#       --lr_strategy=piecewise_decay \
#	--num_epochs=120 \
#       --lr=0.1

#ResNet101:
#python train.py \
#       --model=ResNet101 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=False \
#       --lr_strategy=piecewise_decay \
#	--num_epochs=120 \
#       --lr=0.1

