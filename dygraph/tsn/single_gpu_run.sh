configs=$1
pretrain="./ResNet50_pretrained/" # set pretrain model path if needed
resume="" # set checkpoints model path if u want to resume training
save_dir=""
use_gpu=True
use_data_parallel=False
weights="" #set the path of weights to enable eval and predicut, just ignore this when training

export CUDA_VISIBLE_DEVICES=0


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
