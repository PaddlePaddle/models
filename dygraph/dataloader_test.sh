#!/bin/bash
mkdir dataloader_test_log

# single card
python ./mnist/train.py > dataloader_test_log/mnist 2>&1
python ./resnet/train.py > dataloader_test_log/resnet 2>&1
python ./se_resnet/train.py > dataloader_test_log/se_resnet 2>&1
python ./transformer/train.py > dataloader_test_log/transformer 2>&1

python ./mnist/train_sp.py > dataloader_test_log/mnist_sp 2>&1
python ./resnet/train_sp.py > dataloader_test_log/resnet_sp 2>&1
python ./se_resnet/train_sp.py > dataloader_test_log/se_resnet_sp 2>&1
python ./transformer/train_sp.py > dataloader_test_log/transformer_sp 2>&1

python ./mnist/train_mp.py > dataloader_test_log/mnist_mp 2>&1
python ./resnet/train_mp.py > dataloader_test_log/resnet_mp 2>&1
python ./se_resnet/train_mp.py > dataloader_test_log/se_resnet_mp 2>&1
python ./transformer/train_mp.py > dataloader_test_log/transformer_mp 2>&1

# multiple card
python -m paddle.distributed.launch --log_dir ./dataloader_test_log/mnist_8 ./mnist/train.py --use_data_parallel 1
python -m paddle.distributed.launch --log_dir ./dataloader_test_log/resnet_8 ./resnet/train.py --use_data_parallel 1
python -m paddle.distributed.launch --log_dir ./dataloader_test_log/se_resnet_8 ./se_resnet/train.py --use_data_parallel 1
python -m paddle.distributed.launch --log_dir ./dataloader_test_log/transformer_8 ./transformer/train.py --use_data_parallel 1

python -m paddle.distributed.launch --log_dir ./dataloader_test_log/mnist_8_sp ./mnist/train_sp.py --use_data_parallel 1
python -m paddle.distributed.launch --log_dir ./dataloader_test_log/resnet_8_sp ./resnet/train_sp.py --use_data_parallel 1
python -m paddle.distributed.launch --log_dir ./dataloader_test_log/se_resnet_8_sp ./se_resnet/train_sp.py --use_data_parallel 1
python -m paddle.distributed.launch --log_dir ./dataloader_test_log/transformer_8_sp ./transformer/train_sp.py --use_data_parallel 1

python -m paddle.distributed.launch --log_dir ./dataloader_test_log/mnist_8_mp ./mnist/train_mp.py --use_data_parallel 1
python -m paddle.distributed.launch --log_dir ./dataloader_test_log/resnet_8_mp ./resnet/train_mp.py --use_data_parallel 1
python -m paddle.distributed.launch --log_dir ./dataloader_test_log/se_resnet_8_mp ./se_resnet/train_mp.py --use_data_parallel 1
python -m paddle.distributed.launch --log_dir ./dataloader_test_log/transformer_8_mp ./transformer/train_mp.py --use_data_parallel 1