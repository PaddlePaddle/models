configs="tsn.yaml"
pretrain="" # set pretrain model path if needed
resume="" # set checkpoints model path if u want to resume training
save_dir=""
use_gpu=True
use_data_parallel=False
weights="" #set the path of weights to enable eval and predicut, just ignore this when training

export CUDA_VISIBLE_DEVICES=0
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

echo $mode "TSN" $configs  $resume $pretrain
if [ "$resume"x != ""x ]; then
    python train.py --config=$configs \
                    --resume=$resume \
                    --use_gpu=$use_gpu \
                    --use_data_parallel=$use_data_parallel
elif [ "$pretrain"x != ""x ]; then
    python train.py --config=$configs \
                    --pretrain=$pretrain \
                    --use_gpu=$use_gpu \
                    --use_data_parallel=$use_data_parallel
else
    python train.py --config=$configs \
                    --use_gpu=$use_gpu \
                    --use_data_parallel=$use_data_parallel
fi
