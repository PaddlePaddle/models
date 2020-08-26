# examples of running programs:
# bash ./run.sh train CTCN ./configs/ctcn.yaml
# bash ./run.sh eval NEXTVLAD ./configs/nextvlad.yaml
# bash ./run.sh predict NONLOCAL ./cofings/nonlocal.yaml

# mode should be one of [train, eval, predict, inference]
# name should be one of [AttentionCluster, AttentionLSTM, NEXTVLAD, NONLOCAL, TSN, TSM, STNET, CTCN]
# configs should be ./configs/xxx.yaml

mode=$1
configs="./tsn.yaml"
pretrain="" # set pretrain model path if needed
resume="" # set pretrain model path if needed
save_dir=""
use_gpu=True
use_data_parallel=True
weights="" #set the path of weights to enable eval and predicut, just ignore this when training

export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98


if [ "$mode"x == "train"x ]; then
    echo $mode "TSN" $configs  $resume $pretrain
    if [ "$resume"x != ""x ]; then
        python -m paddle.distributed.launch --selected_gpus=0,1,2,3 train.py \
                        --config=$configs \
                        --resume=$resume \
                        --use_gpu=$use_gpu \
                        --use_data_parallel=$use_data_parallel
    elif [ "$pretrain"x != ""x ]; then
        python -m paddle.distributed.launch --selected_gpus=0,1,2,3 train.py \
                        --config=$configs \
                        --pretrain=$pretrain \
                        --use_gpu=$use_gpu \
                        --use_data_parallel=$use_data_parallel
    else
        python -m paddle.distributed.launch --selected_gpus=0,1,2,3 train.py \
                        --config=$configs \
                        --use_gpu=$use_gpu\
                        --use_data_parallel=$use_data_parallel
    fi
elif [ "$mode"x == "eval"x ]; then
    echo $mode $name $configs $weights
    if [ "$weights"x != ""x ]; then
        python eval.py --config=$configs \
                       --weights=$weights \
                       --use_gpu=$use_gpu
    else
        python eval.py --config=$configs \
                       --use_gpu=$use_gpu
    fi
else
    echo "Not implemented mode " $mode
fi
