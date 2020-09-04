configs=$1
weights=$2
use_gpu=True
use_data_parallel=False

export CUDA_VISIBLE_DEVICES=0



echo $mode $configs $weights
if [ "$weights"x != ""x ]; then
    python eval.py --config=$configs \
                    --weights=$weights \
                    --use_gpu=$use_gpu
else
    python eval.py --config=$configs \
                    --use_gpu=$use_gpu
fi
