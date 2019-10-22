export LD_LIBRARY_PATH="/home/vis/chao/local/cudnn-7.1/cuda/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/home/vis/chao/codes/convert2paddle/fluid_1.1.0_for_slurm/nccl_2.3.5/lib:$LD_LIBRARY_PATH

mkdir -p results/paris
mkdir -p results/places

FLAGS_fraction_of_gpu_memory_to_use=0.1 \
CUDA_VISIBLE_DEVICES=3 \
FLAGS_eager_delete_tensor_gb=0.0 \
FLAGS_fast_eager_deletion_mode=1 \
python -u test.py \
--pretrained_model 'pretrained_models/LBAM_ParisStreetView' \
--imgfn 'imgs/paris/*GT*.png' \
--maskfn 'imgs/paris/*mask*.png' \
--resultfn 'results/paris'

FLAGS_fraction_of_gpu_memory_to_use=0.1 \
CUDA_VISIBLE_DEVICES=3 \
FLAGS_eager_delete_tensor_gb=0.0 \
FLAGS_fast_eager_deletion_mode=1 \
python -u test.py \
--pretrained_model 'pretrained_models/LBAM_Places10classes' \
--imgfn 'imgs/places/*GT*.png' \
--maskfn 'imgs/places/*mask*.png' \
--resultfn 'results/places'

