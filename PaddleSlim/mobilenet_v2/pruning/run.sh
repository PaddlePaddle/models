export CUDA_VISIBLE_DEVICES=1

nohup python compress.py \
--use_gpu 0 \
--batch_size 1 \
--pretrained_model ./pretrain/MobileNetV1_pretrained \
--config_file "./filter_pruning_uniform.yaml" \
> run.log 2>&1 &

tailf run.log
