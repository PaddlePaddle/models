export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_fraction_of_gpu_memory_to_use=0.01
CUDA_VISIBLE_DEVICES=0 python train.py --model_net SPADE --dataset cityscapes --train_list ./data/cityscapes/train_list --test_list ./data/cityscapes/val_list --crop_type Random --batch_size 1 --epoch 200 --load_height 612 --load_width 1124 --crop_height 512 --crop_width 1024 --label_nc 36
