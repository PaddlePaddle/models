configs="./multi_tsn.yaml"
pretrain="" # set pretrain model path if needed
resume="" # set checkpoints model path if u want to resume training
save_dir=""
use_gpu=True
use_data_parallel=True

export CUDA_VISIBLE_DEVICES=4,5,6,7


echo $mode "TSN" $configs  $resume $pretrain
if [ "$resume"x != ""x ]; then
    python -m paddle.distributed.launch train.py \
                    --config=$configs \
                    --resume=$resume \
                    --use_gpu=$use_gpu \
                    --use_data_parallel=$use_data_parallel
elif [ "$pretrain"x != ""x ]; then
    python -m paddle.distributed.launch train.py \
                    --config=$configs \
                    --pretrain=$pretrain \
                    --use_gpu=$use_gpu \
                    --use_data_parallel=$use_data_parallel
else
    python -m paddle.distributed.launch train.py \
                    --config=$configs \
                    --use_gpu=$use_gpu\
                    --use_data_parallel=$use_data_parallel
fi
