
# for distillation
#-----------------
export CUDA_VISIBLE_DEVICES=0
python compress.py \
--model "MobileNet" \
--teacher_model "ResNet50" \
--teacher_pretrained_model ./data/pretrain/ResNet50_pretrained \
--compress_config ./configs/mobilenetv1_resnet50_distillation.yaml


# for sensitivity filter pruning
#-------------------------------
#export CUDA_VISIBLE_DEVICES=0
#python compress.py \
#--model "MobileNet" \
#--pretrained_model ./data/pretrain/MobileNetV1_pretrained \
#--compress_config ./configs/filter_pruning_sen.yaml

# for uniform filter pruning
#---------------------------
#export CUDA_VISIBLE_DEVICES=2
#python compress.py \
#--model "MobileNet" \
#--pretrained_model ./data/pretrain/MobileNetV1_pretrained \
#--compress_config ./configs/filter_pruning_uniform.yaml

# for quantization
#-----------------
#export CUDA_VISIBLE_DEVICES=2
#python compress.py \
#--batch_size 64 \
#--model "MobileNet" \
#--pretrained_model ./data/pretrain/MobileNetV1_pretrained \
#--compress_config ./configs/quantization.yaml

# for distillation with quantization
#-----------------------------------
#export CUDA_VISIBLE_DEVICES=0
#python compress.py \
#--model "MobileNet" \
#--teacher_model "ResNet50" \
#--teacher_pretrained_model ./data/pretrain/ResNet50_pretrained \
#--compress_config ./configs/quantization_dist.yaml


# for uniform filter pruning with quantization
#---------------------------------------------
#export CUDA_VISIBLE_DEVICES=0
#python compress.py \
#--model "MobileNet" \
#--pretrained_model ./data/pretrain/MobileNetV1_pretrained \
#--compress_config ./configs/quantization_pruning.yaml

