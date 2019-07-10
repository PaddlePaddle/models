export CUDA_VISIBLE_DEVICES=0
. set_env.sh
python infer.py \
    --pretrained ./ckpt/ResNeXt101_32x4d_Release/ \
    --img_list ./data/val_list.txt \
    --img_path ./data/val/

