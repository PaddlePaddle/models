##Training details
#GPU: NVIDIA® Tesla® V100 4cards 120epochs 67h
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

#ResNet50:
python train.py \
       --model=ResNet50 \
       --batch_size=256 \
       --model_save_dir=output/ \
       --lr_strategy=piecewise_decay \
       --num_epochs=120 \
       --lr=0.1 \
       --l2_decay=1e-4
