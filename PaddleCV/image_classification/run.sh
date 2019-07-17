#Hyperparameters config
#Example: SE_ResNext50_32x4d
python train.py \
       --model=SE_ResNeXt50_32x4d \
       --batch_size=400 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --with_mem_opt=True \
       --lr_strategy=cosine_decay \
       --lr=0.1 \
       --num_epochs=200 \
       --l2_decay=1.2e-4 \
#      >log_SE_ResNeXt50_32x4d.txt 2>&1 &

#SE_154
#python train.py \
#            --model=SE_154_vd \
#            --batch_size=256 \
#            --total_images=1281167 \
#            --image_shape=3,224,224 \
#            --class_dim=1000 \
#            --lr_strategy=cosine_decay \
#            --lr=0.1 \
#            --num_epochs=200 \
#            --with_mem_opt=True \
#            --model_save_dir=output/ \
#            --l2_decay=1e-4 \
#            --use_mixup=True \
#            --use_label_smoothing=True \
#            --label_smoothing_epsilon=0.1 \

#ResNeXt101_64x4d
#python train.py \
#	--model=ResNeXt101_64x4d \
#            --batch_size=256 \
#            --total_images=1281167 \
#            --image_shape=3,224,224 \
#            --class_dim=1000 \
#            --lr_strategy=piecewise_decay \
#            --lr=0.1 \
#            --num_epochs=120 \
#            --with_mem_opt=True \
#            --model_save_dir=output/ \
#            --l2_decay=15e-5

#python train.py \
#ResNeXt101_vd_64x4d
#	--model=ResNeXt101_vd_64x4d \
#            --batch_size=256 \
#            --total_images=1281167 \
#            --image_shape=3,224,224 \
#            --class_dim=1000 \
#            --lr_strategy=cosine_decay \
#            --lr=0.1 \
#            --num_epochs=200 \
#            --with_mem_opt=True \
#            --model_save_dir=output/ \
#            --l2_decay=1e-4 \
#            --use_mixup=True \
#            --use_label_smoothing=True \
#            --label_smoothing_epsilon=0.1

#InceptionV4
#python train.py
#	    --model=InceptionV4 \
#            --batch_size=256 \
#            --total_images=1281167 \
#            --image_shape=3,299,299 \
#            --class_dim=1000 \
#            --lr_strategy=cosine_decay \
#            --lr=0.045 \
#            --num_epochs=200 \
#            --with_mem_opt=True \
#            --model_save_dir=output/ \
#            --l2_decay=1e-4 \
#            --use_mixup=True \
#            --resize_short_size=320 \
#            --use_label_smoothing=True \
#            --label_smoothing_epsilon=0.1 \
#ResNet152_vd
#python train.py
#            --model=ResNet152_vd \
#            --batch_size=256 \
#            --total_images=1281167 \
#            --image_shape=3,224,224 \
#            --class_dim=1000 \
#            --lr_strategy=cosine_decay \
#            --lr=0.1 \
#            --num_epochs=200 \
#            --with_mem_opt=True \
#            --model_save_dir=output/ \
#            --l2_decay=1e-4 \
#            --use_mixup=True \
#            --use_label_smoothing=True \
#            --label_smoothing_epsilon=0.1

#ResNet200_vd
#python train.py
#            --model=ResNet200_vd \
#            --batch_size=256 \
#            --total_images=1281167 \
#            --image_shape=3,224,224 \
#            --class_dim=1000 \
#            --lr_strategy=cosine_decay \
#            --lr=0.1 \
#            --num_epochs=200 \
#            --with_mem_opt=True \
#            --model_save_dir=output/ \
#            --l2_decay=1e-4 \
#            --use_mixup=True \
#            --use_label_smoothing=True \
#            --label_smoothing_epsilon=0.1

#ResNet50_vd
#python train.py
#            --model=ResNet50_vd \
#            --batch_size=256 \
#            --total_images=1281167 \
#            --image_shape=3,224,224 \
#            --class_dim=1000 \
#            --lr_strategy=cosine_decay \
#            --lr=0.1 \
#            --num_epochs=200 \
#            --with_mem_opt=True \
#            --model_save_dir=output/ \
#            --l2_decay=7e-5 \
#            --use_mixup=True \
#            --use_label_smoothing=True \
#            --label_smoothing_epsilon=0.1
#ResNet50_vc
#python train.py
#  	    --model=ResNet50_vc \
#	    --batch_size=256 \
#            --total_images=1281167 \
#            --image_shape=3,224,224 \
#            --class_dim=1000 \
#            --lr_strategy=cosine_decay \
#            --lr=0.1 \
#            --num_epochs=200 \
#            --with_mem_opt=True \
#            --model_save_dir=output/ \
#            --l2_decay=1e-4 \

#AlexNet:
#python train.py \
#       --model=AlexNet \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=piecewise_decay \
#	--num_epochs=120 \
#       --lr=0.01 \
#       --l2_decay=1e-4

#MobileNet v1:
#python train.py \
#       --model=MobileNet \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=piecewise_decay \
#	--num_epochs=120 \
#       --lr=0.1 \
#       --l2_decay=3e-5

#python train.py \
#	--model=MobileNetV2 \
#	--batch_size=500 \
#	--total_images=1281167 \
#	--class_dim=1000 \
#	--image_shape=3,224,224 \
#	--model_save_dir=output/ \
#	--with_mem_opt=True \
#	--lr_strategy=cosine_decay \
#	--num_epochs=240 \
#	--lr=0.1 \
#       --l2_decay=4e-5
#ResNet18:
#python train.py \
#	--model=ResNet18 \
#	--batch_size=256 \
#	--total_images=1281167 \
#	--class_dim=1000 \
#	--image_shape=3,224,224 \
#	--model_save_dir=output/ \
#	--with_mem_opt=True \
#	--lr_strategy=cosine_decay \
#	--lr=0.1 \
#	--num_epochs=120 \
#	--l2_decay=1e-4
#ResNet34:
#python train.py \
#	--model=ResNet34 \
#	--batch_size=256 \
#	--total_images=1281167 \
#	--class_dim=1000 \
#	--image_shape=3,224,224 \
#	--model_save_dir=output/ \
#	--with_mem_opt=True \
#	--lr_strategy=cosine_decay \
#	--lr=0.1 \
#	--num_epochs=120 \
#	--l2_decay=1e-4
#ShuffleNetv2:
#python train.py \
#	--model=ShuffleNetV2 \
#	--batch_size=1024 \
#	--total_images=1281167 \
#	--class_dim=1000 \
#	--image_shape=3,224,224 \
#	--model_save_dir=output/ \
#	--with_mem_opt=True \
#	--lr_strategy=cosine_decay_with_warmup \
#	--lr=0.5 \
#	--num_epochs=240 \
#	--l2_decay=4e-5 
#GoogleNet:
#python train.py \
#	--model=GoogleNet \
#	--batch_size=256 \
#	--total_images=1281167 \
#	--class_dim=1000 \
#	--image_shape=3,224,224 \
#	--model_save_dir=output/ \
#	--with_mem_opt=True \
#	--lr_strategy=cosine_decay \
#	--lr=0.01 \
#	--num_epochs=200 \
#	--l2_decay=1e-4
#ResNet50:
#python train.py \
#       --model=ResNet50 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=piecewise_decay \
#	--num_epochs=120 \
#       --lr=0.1 \
#       --l2_decay=1e-4

#ResNet101:
#python train.py \
#       --model=ResNet101 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=piecewise_decay \
#	--num_epochs=120 \
#       --lr=0.1 \
#       --l2_decay=1e-4

#ResNet152:
#python train.py \
#       --model=ResNet152 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --lr_strategy=piecewise_decay \
#       --with_mem_opt=True \
#       --lr=0.1 \
#       --num_epochs=120 \
#       --l2_decay=1e-4


#SE_ResNeXt50_32x4d:
#python train.py \
#       --model=SE_ResNeXt50_32x4d \
#       --batch_size=400 \
#       --total_images=1281167 \
#	--class_dim=1000 \
#       --image_shape=3,224,224 \
#       --lr_strategy=cosine_decay \
#       --model_save_dir=output/ \
#       --lr=0.1 \
#       --num_epochs=200 \
#       --with_mem_opt=True \
#       --l2_decay=1.2e-4

#SE_ResNeXt101_32x4d:
#python train.py \
#        --model=SE_ResNeXt101_32x4d \
#        --batch_size=400 \
#        --total_images=1281167 \
#        --class_dim=1000 \
#        --image_shape=3,224,224 \
#        --lr_strategy=cosine_decay \
#        --model_save_dir=output/ \
#        --lr=0.1 \
#        --num_epochs=200 \
#        --with_mem_opt=True \
#        --l2_decay=1.5e-5

#VGG11:
#python train.py \
#        --model=VGG11 \
#        --batch_size=512 \
#        --total_images=1281167 \
#        --image_shape=3,224,224 \
#        --lr_strategy=cosine_decay \
#        --class_dim=1000 \
#        --model_save_dir=output/ \
#        --lr=0.1 \
#        --num_epochs=90 \
#        --with_mem_opt=True \
#        --l2_decay=2e-4

#VGG13:
#python train.py
#        --model=VGG13 \          
#        --batch_size=256 \
#        --total_images=1281167 \
#        --class_dim=1000 \
#        --image_shape=3,224,224 \
#        --lr_strategy=cosine_decay \
#        --lr=0.01 \
#        --num_epochs=90 \
#        --model_save_dir=output/ \
#        --with_mem_opt=True \
#        --l2_decay=3e-4

#VGG16:
#python train.py
#	--model=VGG16 \
#	--batch_size=256 \
#	--total_images=1281167 \
#       --class_dim=1000 \
#	--lr_strategy=cosine_decay \
#	--image_shape=3,224,224 \
#       --model_save_dir=output/ \
#	--lr=0.01 \
#	--num_epochs=90 \
#       --with_mem_opt=True \
#	--l2_decay=3e-4

#VGG19:
#python train.py
#	--model=VGG19 \
#	--batch_size=256 \
#	--total_images=1281167 \
#	--class_dim=1000 \
#	--image_shape=3,224,224 \
#	--lr_strategy=cosine_decay \
#	--lr=0.01 \
#	--num_epochs=90 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#	--l2_decay=3e-4

#ResNet50 nGraph:
# Training:
#OMP_NUM_THREADS=`nproc` FLAGS_use_ngraph=true python train.py \
#    --model=ResNet50 \
#    --batch_size=128 \
#    --total_images=1281167 \
#    --class_dim=1000 \
#    --image_shape=3,224,224 \
#    --lr=0.001 \
#    --num_epochs=120 \
#    --with_mem_opt=False \
#    --model_save_dir=output/ \
#    --lr_strategy=adam \
#    --use_gpu=False
# Inference:
#OMP_NUM_THREADS=`nproc` FLAGS_use_ngraph=true python infer.py  \
#    --use_gpu=false \
#    --model=ResNet50 \
#    --pretrained_model=ResNet50_pretrained

