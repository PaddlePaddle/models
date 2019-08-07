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
#       --num_epochs=120 \
#       --lr=0.01 \
#       --l2_decay=1e-4

#SqueezeNet1_0
#python train.py \
#        --model=SqueezeNet1_0 \
#        --batch_size=256 \
#        --total_images=1281167 \
#        --image_shape=3,224,224 \
#        --lr_strategy=cosine_decay \
#        --class_dim=1000 \
#        --model_save_dir=output/ \
#        --lr=0.02 \
#        --num_epochs=120 \
#        --with_mem_opt=True \
#        --l2_decay=1e-4

#SqueezeNet1_1
#python train.py \
#        --model=SqueezeNet1_1 \
#        --batch_size=256 \
#        --total_images=1281167 \
#        --image_shape=3,224,224 \
#        --lr_strategy=cosine_decay \
#        --class_dim=1000 \
#        --model_save_dir=output/ \
#        --lr=0.02 \
#        --num_epochs=120 \
#        --with_mem_opt=True \
#        --l2_decay=1e-4

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
#        --model=VGG16 \
#        --batch_size=256 \
#        --total_images=1281167 \
#        --class_dim=1000 \
#        --lr_strategy=cosine_decay \
#        --image_shape=3,224,224 \
#        --model_save_dir=output/ \
#        --lr=0.01 \
#        --num_epochs=90 \
#        --with_mem_opt=True \
#        --l2_decay=3e-4

#VGG19:
#python train.py
#        --model=VGG19 \
#        --batch_size=256 \
#        --total_images=1281167 \
#        --class_dim=1000 \
#        --image_shape=3,224,224 \
#        --lr_strategy=cosine_decay \
#        --lr=0.01 \
#        --num_epochs=90 \
#        --with_mem_opt=True \
#        --model_save_dir=output/ \
#        --l2_decay=3e-4

#MobileNetV1:
#python train.py \
#       --model=MobileNet \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=piecewise_decay \
#       --num_epochs=120 \
#       --lr=0.1 \
#       --l2_decay=3e-5

#MobileNetV2_x0_25
#python train.py \
#       --model=MobileNetV2_x0_25 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_decay \
#       --num_epochs=240 \
#       --lr=0.1 \
#       --l2_decay=3e-5 \
#       --lower_ratio=1.0 \
#       --upper_ratio=1.0

#MobileNetV2_x0_5
#python train.py \
#       --model=MobileNetV2_x0_5 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_decay \
#       --num_epochs=240 \
#       --lr=0.1 \
#       --l2_decay=3e-5 \
#       --lower_ratio=1.0 \
#       --upper_ratio=1.0

#MobileNetV2_x1_0:
#python train.py \
#       --model=MobileNetV2_x1_0 \
#       --batch_size=500 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_decay \
#       --num_epochs=240 \
#       --lr=0.1 \
#       --l2_decay=4e-5

#MobileNetV2_x1_5
#python train.py \
#       --model=MobileNetV2_x1_5 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_decay \
#       --num_epochs=240 \
#       --lr=0.1 \
#       --l2_decay=4e-5 

#MobileNetV2_x2_0
#python train.py \
#       --model=MobileNetV2_x2_0 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_decay \
#       --num_epochs=240 \
#       --lr=0.1 \
#       --l2_decay=4e-5 

#ShuffleNetV2_x0_25:
#python train.py \
#       --model=ShuffleNetV2_x0_25 \
#       --batch_size=1024 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_warmup_decay \
#       --num_epochs=240 \
#       --lr=0.5 \
#       --l2_decay=3e-5 \
#       --lower_scale=0.64 \
#       --lower_ratio=0.8 \
#       --upper_ratio=1.2

#ShuffleNetV2_x0_33:
#python train.py \
#       --model=ShuffleNetV2_x0_33 \
#       --batch_size=1024 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_warmup_decay \
#       --num_epochs=240 \
#       --lr=0.5 \
#       --l2_decay=3e-5 \
#       --lower_scale=0.64 \
#       --lower_ratio=0.8 \
#       --upper_ratio=1.2

#ShuffleNetV2_x0_5:
#python train.py \
#       --model=ShuffleNetV2_x0_5 \
#       --batch_size=1024 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_warmup_decay \
#       --num_epochs=240 \
#       --lr=0.5 \
#       --l2_decay=3e-5 \
#       --lower_scale=0.64 \
#       --lower_ratio=0.8 \
#       --upper_ratio=1.2

#ShuffleNetV2_x1_0:
#python train.py \
#       --model=ShuffleNetV2_x1_0 \
#       --batch_size=1024 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_warmup_decay \
#       --num_epochs=240 \
#       --lr=0.5 \
#       --l2_decay=4e-5 \
#       --lower_scale=0.2

#ShuffleNetV2_x1_5:
#python train.py \
#       --model=ShuffleNetV2_x1_5 \
#       --batch_size=512 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_warmup_decay \
#       --num_epochs=240 \
#       --lr=0.25 \
#       --l2_decay=4e-5 \
#       --lower_ratio=1.0 \
#       --upper_ratio=1.0

#ShuffleNetV2_x2_0:
#python train.py \
#       --model=ShuffleNetV2_x2_0 \
#       --batch_size=512 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_warmup_decay \
#       --num_epochs=240 \
#       --lr=0.25 \
#       --l2_decay=4e-5 

#ShuffleNetV2_x1_0_swish:
#python train.py \
#       --model=ShuffleNetV2_x1_0_swish \
#       --batch_size=1024 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_warmup_decay \
#       --lr=0.5 \
#       --num_epochs=240 \
#       --l2_decay=4e-5 

#ResNet18:
#python train.py \
#       --model=ResNet18 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_decay \
#       --lr=0.1 \
#       --num_epochs=120 \
#       --l2_decay=1e-4

#ResNet34:
#python train.py \
#       --model=ResNet34 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_decay \
#       --lr=0.1 \
#       --num_epochs=120 \
#       --l2_decay=1e-4

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
#       --num_epochs=120 \
#       --lr=0.1 \
#       --l2_decay=1e-4

#ResNet50_vc
#python train.py
#       --model=ResNet50_vc \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=cosine_decay \
#       --lr=0.1 \
#       --num_epochs=200 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4 \

#ResNet50_vd
#python train.py
#       --model=ResNet50_vd \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=cosine_decay \
#       --lr=0.1 \
#       --num_epochs=200 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=7e-5 \
#       --use_mixup=True \
#       --use_label_smoothing=True \
#       --label_smoothing_epsilon=0.1

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
#       --num_epochs=120 \
#       --lr=0.1 \
#       --l2_decay=1e-4

#ResNet101_vd
#python train.py
#       --model=ResNet101_vd \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=cosine_decay \
#       --lr=0.1 \
#       --num_epochs=200 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4 \
#       --use_mixup=True \
#       --use_label_smoothing=True \
#       --label_smoothing_epsilon=0.1

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

#ResNet152_vd
#python train.py
#       --model=ResNet152_vd \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=cosine_decay \
#       --lr=0.1 \
#       --num_epochs=200 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4 \
#       --use_mixup=True \
#       --use_label_smoothing=True \
#       --label_smoothing_epsilon=0.1

#ResNet200_vd
#python train.py
#       --model=ResNet200_vd \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=cosine_decay \
#       --lr=0.1 \
#       --num_epochs=200 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4 \
#       --use_mixup=True \
#       --use_label_smoothing=True \
#       --label_smoothing_epsilon=0.1

#ResNeXt50_32x4d
#python train.py \
#       --model=ResNeXt50_32x4d \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=piecewise_decay \
#       --lr=0.1 \
#       --num_epochs=120 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4

#ResNeXt50_vd_32x4d
#python train.py \
#       --model=ResNeXt50_vd_32x4d \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=cosine_decay \
#       --lr=0.1 \
#       --num_epochs=200 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4 \
#       --use_mixup=True \
#       --use_label_smoothing=True \
#       --label_smoothing_epsilon=0.1 \

#ResNeXt50_64x4d
#python train.py \
#       --model=ResNeXt50_64x4d \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=piecewise_decay \
#       --lr=0.1 \
#       --num_epochs=120 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4

#ResNeXt50_vd_64x4d
#python train.py \
#       --model=ResNeXt50_vd_64x4d \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=cosine_decay \
#       --lr=0.1 \
#       --num_epochs=200 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4 \
#       --use_mixup=True \
#       --use_label_smoothing=True \
#       --label_smoothing_epsilon=0.1 \

#ResNeXt101_32x4d
#python train.py \
#       --model=ResNeXt101_32x4d \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=piecewise_decay \
#       --lr=0.1 \
#       --num_epochs=120 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4

#ResNeXt101_64x4d
#python train.py \
#       --model=ResNeXt101_64x4d \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=piecewise_decay \
#       --lr=0.1 \
#       --num_epochs=120 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=15e-5

#ResNeXt101_vd_64x4d
# python train.py \
#       --model=ResNeXt101_vd_64x4d \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=cosine_decay \
#       --lr=0.1 \
#       --num_epochs=200 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4 \
#       --use_mixup=True \
#       --use_label_smoothing=True \
#       --label_smoothing_epsilon=0.1

# ResNeXt152_32x4d
# python train.py \
#       --model=ResNeXt152_32x4d \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=piecewise_decay \
#       --lr=0.1 \
#       --num_epochs=120 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4

#ResNeXt152_64x4d
#python train.py \
#       --model=ResNeXt152_64x4d \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=piecewise_decay \
#       --lr=0.1 \
#       --num_epochs=120 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=18e-5

# DenseNet121
# python train.py \
#       --model=DenseNet121 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=piecewise_decay \
#       --lr=0.1 \
#       --num_epochs=120 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4

# DenseNet161
# python train.py \
#       --model=DenseNet161 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=piecewise_decay \
#       --lr=0.1 \
#       --num_epochs=120 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4

# DenseNet169
# python train.py \
#       --model=DenseNet169 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=piecewise_decay \
#       --lr=0.1 \
#       --num_epochs=120 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4

# DenseNet201
# python train.py \
#       --model=DenseNet201 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=piecewise_decay \
#       --lr=0.1 \
#       --num_epochs=120 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4

# DenseNet264
# python train.py \
#       --model=DenseNet264 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=piecewise_decay \
#       --lr=0.1 \
#       --num_epochs=120 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4

#SE_ResNeXt50_32x4d:
#python train.py \
#       --model=SE_ResNeXt50_32x4d \
#       --batch_size=400 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --lr_strategy=cosine_decay \
#       --model_save_dir=output/ \
#       --lr=0.1 \
#       --num_epochs=200 \
#       --with_mem_opt=True \
#       --l2_decay=1.2e-4

#SE_ResNeXt101_32x4d:
#python train.py \
#       --model=SE_ResNeXt101_32x4d \
#       --batch_size=400 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --lr_strategy=cosine_decay \
#       --model_save_dir=output/ \
#       --lr=0.1 \
#       --num_epochs=200 \
#       --with_mem_opt=True \
#       --l2_decay=1.5e-5

# SE_154
# python train.py \
#       --model=SE_154_vd \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=cosine_decay \
#       --lr=0.1 \
#       --num_epochs=200 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4 \
#       --use_mixup=True \
#       --use_label_smoothing=True \
#       --label_smoothing_epsilon=0.1 \

#GoogleNet:
#python train.py \
#       --model=GoogleNet \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_decay \
#       --lr=0.01 \
#       --num_epochs=200 \
#       --l2_decay=1e-4

# Xception_41
# python train.py \
#       --model=Xception_41 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=cosine_decay \
#       --lr=0.045 \
#       --num_epochs=120 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4 \
#       --resize_short_size=320

# InceptionV4
# python train.py
#       --model=InceptionV4 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,299,299 \
#       --class_dim=1000 \
#       --lr_strategy=cosine_decay \
#       --lr=0.045 \
#       --num_epochs=200 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4 \
#       --use_mixup=True \
#       --resize_short_size=320 \
#       --use_label_smoothing=True \
#       --label_smoothing_epsilon=0.1 \

#DarkNet53
 python train.py
#       --model=DarkNet53 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,256,256 \
#       --class_dim=1000 \
#       --lr_strategy=cosine_decay \
#       --lr=0.1 \
#       --num_epochs=200 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4 \
#       --use_mixup=True \
#       --resize_short_size=256 \
#       --use_label_smoothing=True \
#       --label_smoothing_epsilon=0.1 \

#ResNet50 nGraph:
# Training:
#OMP_NUM_THREADS=`nproc` FLAGS_use_ngraph=true python train.py \
#       --model=ResNet50 \
#       --batch_size=128 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --lr=0.001 \
#       --num_epochs=120 \
#       --with_mem_opt=False \
#       --model_save_dir=output/ \
#       --lr_strategy=adam \
#       --use_gpu=False
# Inference:
#OMP_NUM_THREADS=`nproc` FLAGS_use_ngraph=true python infer.py  \
#       --use_gpu=false \
#       --model=ResNet50 \
#       --pretrained_model=ResNet50_pretrained
