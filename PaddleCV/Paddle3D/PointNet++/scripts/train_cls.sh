export CUDA_VISIBLE_DEVICES=0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python -c 'import paddle; print(paddle.sysconfig.get_lib())'`
python train_cls.py --model=MSG --batch_size=16 --num_points=4096 --epoch=200
