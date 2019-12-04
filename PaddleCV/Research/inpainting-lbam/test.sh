mkdir -p results/paris

FLAGS_fraction_of_gpu_memory_to_use=0.1 \
CUDA_VISIBLE_DEVICES=0 \
FLAGS_eager_delete_tensor_gb=0.0 \
FLAGS_fast_eager_deletion_mode=1 \
python -u test.py \
--pretrained_model 'pretrained_models/LBAM_ParisStreetView' \  # path to the pretrained model
--imgfn 'imgs/paris/pic.png' \                                 # input picture.
--maskfn 'imgs/paris/mask.png' \                               # mask.
--resultfn 'results/paris'                                     # folder for the result.