# examples of running programs:
# bash ./run.sh train CTCN ./configs/ctcn.txt
# bash ./run.sh eval NEXTVLAD ./configs/nextvlad.txt
# bash ./run.sh predict NONLOCAL ./cofings/nonlocal.txt

# mode should be one of [train, test, infer]
# name should be one of [AttentionCluster, AttentionLSTM, NEXTVLAD, TSN, TSM, STNET, CTCN]
# configs should be ./configs/xxx.yaml

mode=$1
name=$2
configs=$3

train_pretrain="" # set pretrain model path if needed
train_checkpoints="" # set pretrain model path if needed

test_weights="" #set the path of weights to enable eval and predict

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

if [ "$mode"x == "train"x ]; then
    echo $mode $name $configs $train_checkpoints $train_pretrain
    if [ "$train_checkpoints"x != ""x ]; then
        python train.py --model_name=$name --config=$configs --log_interval=1 --valid_interval=1 \
                           --checkpoints=$train_checkpoints
    elif [ "$train_pretrain"x != ""x ]; then
        python train.py --model_name=$name --config=$configs --log_interval=1 --valid_interval=1 \
                           --pretrain=$train_pretrain
    else
        python train.py --model_name=$name --config=$configs --log_interval=1 --valid_interval=1
    fi
elif [ "$mode"x == "eval"x ]; then
    echo $mode $name $configs $test_weights
    python eval.py --model_name=$name --config=$configs --log_interval=1 \
                   --weights=$test_weights
elif [ "$mode"x == "predict"x ]; then
    echo $mode $name $configs $test_weights
    python predict.py --model_name=$name --config=$configs --log_interval=1 \
                      --weights=$test_weights
else
    echo "Not implemented mode " $mode
fi
