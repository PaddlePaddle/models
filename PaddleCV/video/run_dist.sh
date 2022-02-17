# examples of running programs:
# bash ./run.sh train CTCN ./configs/ctcn.yaml
# bash ./run.sh eval NEXTVLAD ./configs/nextvlad.yaml
# bash ./run.sh predict NONLOCAL ./cofings/nonlocal.yaml

# mode should be one of [train, eval, predict, inference]
# name should be one of [AttentionCluster, AttentionLSTM, NEXTVLAD, NONLOCAL, TSN, TSM, STNET, CTCN]
# configs should be ./configs/xxx.yaml

mode=$1
name=$2
configs=$3

pretrain="" # set pretrain model path if needed
resume="" # set pretrain model path if needed
save_dir="./data/checkpoints"
save_inference_dir="./data/inference_model"
use_gpu=True
fix_random_seed=False
log_interval=1
valid_interval=1

weights="" #set the path of weights to enable eval and predicut, just ignore this when training

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=0
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

if [ "$mode"x == "train"x ]; then
    echo $mode $name $configs $resume $pretrain
    if [ "$resume"x != ""x ]; then
        python -m paddle.distributed.launch --log_dir=log \
                train_dist.py --model_name=$name \
                        --config=$configs \
                        --resume=$resume \
                        --log_interval=$log_interval \
                        --valid_interval=$valid_interval \
                        --use_gpu=$use_gpu \
                        --save_dir=$save_dir \
                        --fix_random_seed=$fix_random_seed
    elif [ "$pretrain"x != ""x ]; then
        python -m paddle.distributed.launch --log_dir=log \
                train_dist.py --model_name=$name \
                        --config=$configs \
                        --pretrain=$pretrain \
                        --log_interval=$log_interval \
                        --valid_interval=$valid_interval \
                        --use_gpu=$use_gpu \
                        --save_dir=$save_dir \
                        --fix_random_seed=$fix_random_seed
    else
        python -m paddle.distributed.launch --log_dir=log \
                train_dist.py --model_name=$name \
                        --config=$configs \
                        --log_interval=$log_interval \
                        --valid_interval=$valid_interval \
                        --use_gpu=$use_gpu \
                        --save_dir=$save_dir \
                        --fix_random_seed=$fix_random_seed

    fi
else
    echo "Not implemented mode " $mode
fi

