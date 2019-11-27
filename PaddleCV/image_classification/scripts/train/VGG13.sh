#Training details
#GPU: NVIDIA® Tesla® V100 4cards 90epochs 58h
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

#VGG13:
python train.py \
        --model=VGG13 \
        --batch_size=256 \
        --lr_strategy=cosine_decay \
        --lr=0.01 \
        --num_epochs=90 \
        --model_save_dir=output/ \
        --l2_decay=3e-4
