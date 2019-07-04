export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=1.0

python train.py --model_name="CTCN" --config=./configs/ctcn.txt --epoch=35 \
                --valid_interval=1 --log_interval=1
