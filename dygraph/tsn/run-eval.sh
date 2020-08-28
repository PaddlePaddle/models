configs="tsn-test.yaml"
use_gpu=True
use_data_parallel=False

export CUDA_VISIBLE_DEVICES=0
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98


echo $mode $configs $weights
if [ "$weights"x != ""x ]; then
    python eval.py --config=$configs \
                    --weights=$weights \
                    --use_gpu=$use_gpu
else
    python eval.py --config=$configs \
                    --use_gpu=$use_gpu
fi
