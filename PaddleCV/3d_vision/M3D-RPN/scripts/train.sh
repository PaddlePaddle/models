export CUDA_VISIBLE_DEVICES=1
export export FLAGS_fraction_of_gpu_memory_to_use=0.1
python train.py --data_dir dataset --conf kitti_3d_multi_warmup
