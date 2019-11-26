#!/usr/bin/env bash

# download pretrain model
root_url="http://paddle-imagenet-models-name.bj.bcebos.com"
ResNet50="ResNet50_pretrained.tar"
pretrain_dir='../pretrain'

if [ ! -d ${pretrain_dir} ]; then
  mkdir ${pretrain_dir}
fi

cd ${pretrain_dir}

if [ ! -f ${ResNet50} ]; then
    wget ${root_url}/${ResNet50}
    tar xf ${ResNet50}
fi

cd -

# enable GC strategy
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0

# for distillation
#-----------------
export CUDA_VISIBLE_DEVICES=0,1,2,3

# for mobilenet_v1 distillation
cd ${pretrain_dir}/ResNet50_pretrained
for files in $(ls res50_*)
    do mv $files ${files#*_}
done
for files in $(ls *)
    do mv $files "res50_"$files
done
cd -

python -u compress.py \
--model "MobileNet" \
--teacher_model "ResNet50" \
--teacher_pretrained_model ../pretrain/ResNet50_pretrained \
--compress_config ./configs/mobilenetv1_resnet50_distillation.yaml \
> mobilenet_v1.log 2>&1 &
tailf mobilenet_v1.log

## for mobilenet_v2 distillation
#cd ${pretrain_dir}/ResNet50_pretrained
#for files in $(ls res50_*)
#    do mv $files ${files#*_}
#done
#for files in $(ls *)
#    do mv $files "res50_"$files
#done
#cd -
#
#python -u compress.py \
#--model "MobileNetV2" \
#--teacher_model "ResNet50" \
#--teacher_pretrained_model ../pretrain/ResNet50_pretrained \
#--compress_config ./configs/mobilenetv2_resnet50_distillation.yaml\
#> mobilenet_v2.log 2>&1 &
#tailf mobilenet_v2.log

## for resnet34 distillation
#cd ${pretrain_dir}/ResNet50_pretrained
#for files in $(ls res50_*)
#    do mv $files ${files#*_}
#done
#for files in $(ls *)
#    do mv $files "res50_"$files
#done
#cd -
#
#python -u compress.py \
#--model "ResNet34" \
#--teacher_model "ResNet50" \
#--teacher_pretrained_model ../pretrain/ResNet50_pretrained \
#--compress_config ./configs/resnet34_resnet50_distillation.yaml \
#> resnet34.log 2>&1 &
#tailf resnet34.log
