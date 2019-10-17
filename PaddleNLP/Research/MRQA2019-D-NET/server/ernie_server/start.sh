export FLAGS_fraction_of_gpu_memory_to_use=0.1
port=$1
gpu=$2
export CUDA_VISIBLE_DEVICES=$gpu
python start_service.py ./infer_model $port

