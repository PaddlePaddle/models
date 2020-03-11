export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.96


python -u train.py \
       --model=EfficientNet \
       --batch_size=512 \
       --test_batch_size=128 \
       --resize_short_size=256 \
       --model_save_dir=output/ \
       --lr_strategy=exponential_decay_warmup \
       --lr=0.032 \
       --num_epochs=360 \
       --l2_decay=1e-5 \
       --use_label_smoothing=True \
       --label_smoothing_epsilon=0.1 \
       --use_ema=True \
       --ema_decay=0.9999 \
       --drop_connect_rate=0.1 \
       --padding_type="SAME" \
       --interpolation=2 \
       --use_aa=True
