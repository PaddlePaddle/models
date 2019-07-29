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

pretrain="" # set pretrain model path if needed
resume="" # set pretrain model path if needed

weights="./checkpoints/CTCN_final.pdparams" #set the path of weights to enable eval and predicut, just ignore this when training

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=0,5,6,7
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

if [ "$mode"x == "train"x ]; then
    echo $mode $name $configs $resume $pretrain
    if [ "$resume"x != ""x ]; then
        python train.py --model_name=$name --config=$configs --log_interval=1 --valid_interval=1 \
                           --resume=$resume
    elif [ "$pretrain"x != ""x ]; then
        python train.py --model_name=$name --config=$configs --log_interval=1 --valid_interval=1 \
                           --pretrain=$pretrain
    else
        python train.py --model_name=$name --config=$configs --log_interval=1 --valid_interval=1
    fi
elif [ "$mode"x == "eval"x ]; then
    echo $mode $name $configs $weights
    python eval.py --model_name=$name --config=$configs --log_interval=1 \
                   --weights=$weights
elif [ "$mode"x == "predict"x ]; then
    echo $mode $name $configs $weights
    python predict.py --model_name=$name --config=$configs --log_interval=1 \
                      --weights=$weights \
                      --video_path=/ssd3/sungaofeng/docker/dockermount/data/k400/mp4/Kinetics_trimmed_processed_train/abseiling/IRX7GTz-89Y.mp4
else
    echo "Not implemented mode " $mode
fi
