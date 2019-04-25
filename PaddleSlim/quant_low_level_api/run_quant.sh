#!/usr/bin/env bash

# download pretrain model
root_url="http://paddle-imagenet-models-name.bj.bcebos.com"
MobileNetV1="MobileNetV1_pretrained.zip"
ResNet50="ResNet50_pretrained.zip"
GoogleNet="GoogleNet_pretrained.tar"
data_dir='Your image dataset path, e.g. ILSVRC2012'
pretrain_dir='../pretrain'

if [ ! -d ${pretrain_dir} ]; then
  mkdir ${pretrain_dir}
fi

cd ${pretrain_dir}

if [ ! -f ${MobileNetV1} ]; then
    wget ${root_url}/${MobileNetV1}
    unzip ${MobileNetV1}
fi

if [ ! -f ${ResNet50} ]; then
    wget ${root_url}/${ResNet50}
    unzip ${ResNet50}
fi

if [ ! -f ${GoogleNet} ]; then
    wget ${root_url}/${GoogleNet}
    tar xf ${GoogleNet}
fi

cd -


export CUDA_VISIBLE_DEVICES=0,1,2,3

#MobileNet v1:
python quant.py \
       --model=MobileNet \
       --pretrained_fp32_model=${pretrain_dir}/MobileNetV1_pretrained \
       --use_gpu=True \
       --data_dir=${data_dir} \
       --batch_size=256 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --lr_strategy=piecewise_decay \
       --num_epochs=20 \
       --lr=0.0001 \
       --act_quant_type=abs_max \
       --wt_quant_type=abs_max


#ResNet50:
#python quant.py \
#       --model=ResNet50 \
#       --pretrained_fp32_model=${pretrain_dir}/ResNet50_pretrained \
#       --use_gpu=True \
#       --data_dir=${data_dir} \
#       --batch_size=128 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --lr_strategy=piecewise_decay \
#       --num_epochs=20 \
#       --lr=0.0001 \
#       --act_quant_type=abs_max \
#       --wt_quant_type=abs_max

