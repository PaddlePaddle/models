export CUDA_VISIBLE_DEVICES=6,7
python compress.py \
-c yolov3_mobilenet_v1_voc.yml \
-s yolov3_mobilenet_v1_slim.yaml \
> run.log 2>&1 &

tailf run.log
