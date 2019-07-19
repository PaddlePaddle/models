# activate eager gc to reduce memory use
#export FLAGS_fraction_of_gpu_memory_to_use=1.0
#export FLAGS_fast_eager_deletion_mode=1
#export FLAGS_eager_delete_tensor_gb=0.0
#export FLAGS_limit_of_tmp_allocation=0

export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py --model_name="NEXTVLAD" --config=./configs/nextvlad.txt --epoch=6 \
                --valid_interval=1 --log_interval=10
