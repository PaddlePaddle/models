export CUDA_VISIBLE_DEVICES=2,3
python compress.py \
-c yolov3_mobilenet_v1_voc.yml \
-s yolov3_mobilenet_v1_slim.yaml
