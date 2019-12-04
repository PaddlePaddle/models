##Training details
#GPU: NVIDIA® Tesla® P40 8cards 120epochs 566h
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

#SE_ResNeXt101_32x4d:
python train.py \
        --model=SE_ResNeXt101_32x4d \
        --batch_size=400 \
        --lr_strategy=cosine_decay \
        --model_save_dir=output/ \
        --lr=0.1 \
        --num_epochs=200 \
        --l2_decay=1.5e-5
