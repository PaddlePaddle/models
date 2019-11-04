mkdir -p results/paris

FLAGS_fraction_of_gpu_memory_to_use=0.1 \
CUDA_VISIBLE_DEVICES=0 \
FLAGS_eager_delete_tensor_gb=0.0 \
FLAGS_fast_eager_deletion_mode=1 \
python -u test.py \
--pretrained_model 'pretrained_models/LBAM_ParisStreetView' \
--imgfn 'imgs/paris/pic.png' \
--maskfn 'imgs/paris/mask.png' \
--resultfn 'results/paris'