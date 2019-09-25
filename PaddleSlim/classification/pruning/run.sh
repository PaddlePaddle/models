export CUDA_VISIBLE_DEVICES=0

nohup python compress.py \
--model "MobileNet" \
--use_gpu 0 \
--batch_size 1 \
--pretrained_model ../pretrain/MobileNetV1_pretrained \
--config_file "./configs/mobilenet_v1.yaml"
> mobilenet_v1.log 2>&1 &
tailf mobilenet_v1.log

# for compression of mobilenet_v2
#nohup python compress.py \
#--model "MobileNetV2" \
#--use_gpu 0 \
#--batch_size 1 \
#--pretrained_model ../pretrain/MobileNetV2_pretrained \
#--config_file "./configs/mobilenet_v2.yaml" \
#> mobilenet_v2.log 2>&1 &
#tailf mobilenet_v2.log


# for compression of resnet50
#python compress.py \
#--model "ResNet50" \
#--use_gpu 0 \
#--batch_size 1 \
#--pretrained_model ../pretrain/ResNet50_pretrained \
#--config_file "./configs/resnet50.yaml" \
#> resnet50.log 2>&1 &
#tailf resnet50.log

