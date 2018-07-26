export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
cudaid=${object_detection_cudaid:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

if [ ! -d "/root/.cache/paddle/dataset/pascalvoc" ];then
    mkdir -p /root/.cache/paddle/dataset/pascalvoc
    ./data/pascalvoc/download.sh
    bash ./.move.sh
fi
FLAGS_benchmark=true  python train.py --batch_size=64 --num_passes=2 --for_model_ce=True --data_dir=/root/.cache/paddle/dataset/pascalvoc/ 
