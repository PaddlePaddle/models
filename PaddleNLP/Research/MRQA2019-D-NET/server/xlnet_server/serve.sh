export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1
export FLAGS_fraction_of_gpu_memory_to_use=0.1

python serve.py ./infer_model_800_bs128 5001 &
