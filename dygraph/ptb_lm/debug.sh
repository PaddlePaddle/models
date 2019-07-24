
export CUDA_VISIBLE_DEVICES=0

#export FLAGS_fraction_of_gpu_memory_to_use=0.0
python  ptb_dy.py  --data_path data/simple-examples/data/ \
           --model_type small 

