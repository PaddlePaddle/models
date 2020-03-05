#!/bin/bash
mkdir dataloader_test_log

### Reader using models ###

######## single card ########

python ./mnist/train_base.py > dataloader_test_log/mnist_base 2>&1
python ./resnet/train_base.py > dataloader_test_log/resnet_base 2>&1
python ./se_resnet/train_base.py > dataloader_test_log/se_resnet_base 2>&1
python ./ptb_lm/ptb_dy_base.py > dataloader_test_log/ptb_base 2>&1

python ./mnist/train.py > dataloader_test_log/mnist 2>&1
python ./resnet/train.py > dataloader_test_log/resnet 2>&1
python ./se_resnet/train.py > dataloader_test_log/se_resnet 2>&1
python ./ptb_lm/ptb_dy.py > dataloader_test_log/ptb 2>&1

python ./mnist/train_sp.py > dataloader_test_log/mnist_sp 2>&1
python ./resnet/train_sp.py > dataloader_test_log/resnet_sp 2>&1
python ./se_resnet/train_sp.py > dataloader_test_log/se_resnet_sp 2>&1
python ./ptb_lm/ptb_dy_sp.py > dataloader_test_log/ptb_sp 2>&1

python ./mnist/train_mp.py > dataloader_test_log/mnist_mp 2>&1
python ./resnet/train_mp.py > dataloader_test_log/resnet_mp 2>&1
python ./resnet/train_mp_xmap.py > dataloader_test_log/resnet_mp_xmap 2>&1
python ./se_resnet/train_mp.py > dataloader_test_log/se_resnet_mp 2>&1
python ./ptb_lm/ptb_dy_mp.py > dataloader_test_log/ptb_mp 2>&1

### DataLoader using models ### 

# monilenet v1

cd ./mobilenet

export CUDA_VISIBLE_DEVICES=0

python train.py \
    --batch_size=256 \
    --total_images=1281167 \
    --class_dim=1000 \
    --image_shape=3,224,224 \
    --model_save_dir=output.v1.sing/ \
    --lr_strategy=piecewise_decay \
    --lr=0.1 \
    --data_dir=./data/ILSVRC2012 \
    --l2_decay=3e-5 \
    --model=MobileNetV1 \
    > ../dataloader_test_log/mobilenet_v1_sp 2>&1

python train.py \
    --batch_size=256 \
    --total_images=1281167 \
    --class_dim=1000 \
    --image_shape=3,224,224 \
    --model_save_dir=output.v1.sing/ \
    --lr_strategy=piecewise_decay \
    --lr=0.1 \
    --data_dir=./data/ILSVRC2012 \
    --l2_decay=3e-5 \
    --model=MobileNetV1 \
    --use_multiprocess=True \
    > ../dataloader_test_log/mobilenet_v1_mp 2>&1

###### multiple card ########

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch --log_dir ../dataloader_test_log/mobilenet_4_sp.v1 train.py \
    --use_data_parallel 1 \
    --batch_size=256 \
    --total_images=1281167 \
    --class_dim=1000 \
    --image_shape=3,224,224 \
    --lr_strategy=piecewise_decay \
    --lr=0.1 \
    --data_dir=./data/ILSVRC2012 \
    --l2_decay=3e-5 \
    --model=MobileNetV1 \
    --model_save_dir=output.v1.mul/ \
    --num_epochs=120

python -m paddle.distributed.launch --log_dir ../dataloader_test_log/mobilenet_4_mp.v1 train.py \
    --use_data_parallel 1 \
    --batch_size=256 \
    --total_images=1281167 \
    --class_dim=1000 \
    --image_shape=3,224,224 \
    --lr_strategy=piecewise_decay \
    --lr=0.1 \
    --data_dir=./data/ILSVRC2012 \
    --l2_decay=3e-5 \
    --model=MobileNetV1 \
    --model_save_dir=output.v1.mul/ \
    --num_epochs=120 \
    --use_multiprocess=True

cd ..

# transformer 

# cd ./transformer

# export CUDA_VISIBLE_DEVICES=7

# python -u train.py \
#   --epoch 30 \
#   --src_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
#   --trg_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
#   --special_token '<s>' '<e>' '<unk>' \
#   --training_file gen_data/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de \
#   --batch_size 4096 \
#   > ../dataloader_test_log/transformer_sp 2>&1

# python -u train.py \
#   --epoch 30 \
#   --src_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
#   --trg_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
#   --special_token '<s>' '<e>' '<unk>' \
#   --training_file gen_data/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de \
#   --batch_size 4096 \
#   --use_multiprocess True \
#   > ../dataloader_test_log/transformer_mp 2>&1

# # ###### multiple card ########

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# python -m paddle.distributed.launch --started_port 8999 --selected_gpus=0,1,2,3,4,5,6,7 --log_dir ../dataloader_test_log/transformer_8_sp train.py \
#     --epoch 30 \
#     --src_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
#     --trg_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
#     --special_token '<s>' '<e>' '<unk>' \
#     --training_file gen_data/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de \
#     --batch_size 4096 \
#     --print_step 100 \
#     --use_cuda True \
#     --save_step 10000

# python -m paddle.distributed.launch --started_port 8999 --selected_gpus=0,1,2,3,4,5,6,7 --log_dir ../dataloader_test_log/transformer_8_mp train.py \
#     --epoch 30 \
#     --src_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
#     --trg_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
#     --special_token '<s>' '<e>' '<unk>' \
#     --training_file gen_data/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de \
#     --batch_size 4096 \
#     --print_step 100 \
#     --use_cuda True \
#     --save_step 10000 \
#     --use_multiprocess True

# cd ..