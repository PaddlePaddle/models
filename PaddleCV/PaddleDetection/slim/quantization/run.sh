# enable GC strategy
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0

export CUDA_VISIBLE_DEVICES=1,2

python compress.py \
    -s yolov3_mobilenet_v1_slim.yaml \
    -c yolov3_mobilenet_v1_voc.yml \
> yolov3_mobilenet_v1_voc.log 2>&1 & \
tailf yolov3_mobilenet_v1_voc.log
