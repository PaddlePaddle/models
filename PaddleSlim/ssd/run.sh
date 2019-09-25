export CUDA_VISIBLE_DEVICES=2
python train.py \
--dataset="coco2014" \
--data_dir="./data/coco" \
 > ./run.log 2>&1 &


tailf run.log
