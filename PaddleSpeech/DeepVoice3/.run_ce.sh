#!/bin/bash
export FLAGS_fraction_of_gpu_memory_to_use=0.5
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1


train_single_frame()
{
    python train.py \
    --data-root=data/ljspeech/ \
    --use-gpu \
    --preset=presets/deepvoice3_ljspeech.json \
    --hparams="nepochs=10"
}


train_multi_frame()
{
    python train.py \
    --data-root=data/ljspeech/ \
    --use-gpu \
    --preset=presets/deepvoice3_ljspeech.json \
    --hparams="nepochs=10, downsample_step=1, outputs_per_step=4"

}
export CUDA_VISIBLE_DEVICES=0
train_single_frame | python _ce.py
sleep 20
train_multi_frame | python _ce.py

