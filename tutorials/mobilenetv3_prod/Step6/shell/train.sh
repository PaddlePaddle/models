export CUDA_VISIBLE_DEVICES=0

python3.7 train.py \
    --data-path /paddle/data/ILSVRC2012/ \
    --model mobilenet_v3_small \
    --lr 0.1 \
    --batch-size=256 \
    --output-dir "./output/" \
    --epochs 120 \
    --workers=6
