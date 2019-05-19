source ~/zhumanyu/miniconda2/set_env.sh
export LD_LIBRARY_PATH=/usr/local/lib/:~/zhumanyu/cuda-9.0/lib64/:~/zhumanyu/cudnn-7.4/cuda/lib64/:$LD_LIBRARY_PATH
#export FLAGS_reader_queue_speed_test_mode=True
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_fraction_of_gpu_memory_to_use=0.01
CUDA_VISIBLE_DEVICES=1 python train.py --model_net Pix2pix --dataset cityscapes --train_list data/cityscapes/pix2pix_train_list --test_list data/cityscapes/pix2pix_test_list10 --crop_type Random --dropout True --gan_mode vanilla --batch_size 1 > log_out 2>log_err
