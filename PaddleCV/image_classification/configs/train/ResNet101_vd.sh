#Training details
#GPU: NVIDIA® Tesla® V100  4cards 1282epochs 55h
#NOTE: all _vd series is distilled version, and use_distill is temporary disabled now.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

#ResNet101_vd
python train.py
       --model=ResNet101_vd \
       --batch_size=256 \
       --total_images=1281167 \
       --image_shape=3,224,224 \
       --class_dim=1000 \
       --lr_strategy=cosine_decay \
       --lr=0.1 \
       --num_epochs=200 \
       --model_save_dir=output/ \
       --l2_decay=1e-4 \
       --use_mixup=True \
       --use_label_smoothing=True \
       --label_smoothing_epsilon=0.1
