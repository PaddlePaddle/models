# enable GC strategy
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0,1,2,3
python search.py
