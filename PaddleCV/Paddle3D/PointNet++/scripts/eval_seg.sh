export CUDA_VISIBLE_DEVICES=0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python -c 'import paddle; print(paddle.sysconfig.get_lib())'`
python eval_seg.py --model=MSG --weights=checkpoints/200
