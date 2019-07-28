export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=1.0
python infer.py \
    --model ResNeXt101_32x4d \
    --class_dim 5000 \
    --pretrained ./ckpt/ResNeXt101_32x4d_Release/ \
    --img_list ./data/val_list.txt \
    --img_path ./data/val/ \
    --use_gpu True

