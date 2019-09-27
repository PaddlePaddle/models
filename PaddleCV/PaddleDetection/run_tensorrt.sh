export CUDA_VISIBLE_DEVICES=0
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_conv_workspace_size_limit=2048
#export FLAGS_fraction_of_gpu_memory_to_use=0.9
#export FLAGS_fraction_of_gpu_memory_to_use=0.1
export PYTHONPATH=`pwd`:$PYTHONPATH
export LD_LIBRARY_PATH=/paddle/work/trt/TensorRT-5.1.5.0/lib/:/paddle/work/trt/cudnnv7.5_cuda9.0/lib64:$LD_LIBRARY_PATH
rm -rf output/000000014439.jpg
rm -rf output/000000014439_640x640.jpg

# SSD on VOC
#CONFIG=ssd_mobilenet_v1_voc
#TRANSFORM="[!DecodeImage {to_rgb: true},!ResizeImage {target_size: 640,interp: 2},!Permute {to_bgr: true,channel_first: true},!NormalizeImage {mean:[127.5,127.5,127.5],std:[127.502231,127.502231,127.502231],is_scale: false, is_channel_first: true}]"
## save model
#python tools/infer.py -c configs/ssd/${CONFIG}.yml --infer_img demo/000000014439.jpg -o weights=https://paddlemodels.bj.bcebos.com/object_detection/${CONFIG}.tar SSDTestFeed.image_shape=[3,640,640] SSDTestFeed.sample_transforms="${TRANSFORM}" --save_inference_model
## Fluid
#python tools/tensorrt.py -c configs/ssd/${CONFIG}.yml --model_path=output/${CONFIG} -o SSDTestFeed.image_shape=[3,640,640] SSDTestFeed.batch_size=1 SSDTestFeed.sample_transforms="${TRANSFORM}" --mode=fluid --visualize --infer_img demo/000000014439_640x640.jpg #--is_eval #--visualize --is_eval
## TensorRT
#python tools/tensorrt.py -c configs/ssd/${CONFIG}.yml --model_path=output/${CONFIG} -o SSDTestFeed.image_shape=[3,640,640] SSDTestFeed.batch_size=1 SSDTestFeed.sample_transforms="${TRANSFORM}" --mode=trt_fp32 --visualize --infer_img demo/000000014439_640x640.jpg
#

# Yolo on COCO
CONFIG=yolov3_mobilenet_v1 # yolov3_darknet or yolov3_mobilenet_v1 or yolov3_r34
MODEL=https://paddlemodels.bj.bcebos.com/object_detection/${CONFIG}.tar
#TRANSFORM="[!DecodeImage {to_rgb: true},!ResizeImage {target_size: 320,interp: 2},!NormalizeImage {mean: [0.485,0.456,0.406],std:[0.229,0.224,0.225],is_scale: true, is_channel_first: false},!Permute {to_bgr: false}]"
# Save model by 320 x 320
#python tools/infer.py -c configs/${CONFIG}.yml --infer_img demo/000000014439.jpg --save_inference_model \
#    -o weights=${MODEL} YoloTestFeed.image_shape=[3,320,320] YoloTestFeed.sample_transforms="${TRANSFORM}"
# Fluid FP32
#python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/${CONFIG} --mode=fluid \
#    -o YoloTestFeed.batch_size=1 YoloTestFeed.image_shape=[3,640,640] --visualize --infer_img demo/000000014439_640x640.jpg
## TensoRT FP32
#python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/${CONFIG} --mode=trt_fp32 --visualize \
#    -o YoloTestFeed.batch_size=1 YoloTestFeed.image_shape=[3,320,320] --infer_img demo/000000014439.jpg

# FasterRCNN
#CONFIG=faster_rcnn_r50_1x
CONFIG=faster_rcnn_r101_fpn_1x
MODLE=https://paddlemodels.bj.bcebos.com/object_detection/${CONFIG}.tar
# Save model
#python tools/infer.py -c configs/${CONFIG}.yml --infer_img demo/000000014439.jpg -o weights=${MODLE} FasterRCNNTestFeed.image_shape=[3,640,640] --save_inference_model
# TensorRT FP32
#python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/$CONFIG --mode=trt_fp32 --min_subgraph_size=40 --visualize \
#    --infer_img=demo/000000014439_640x640.jpg \
#    -o FasterRCNNTestFeed.batch_size=4
## TensorRT FP16
#python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/$CONFIG --mode=trt_fp16 --min_subgraph_size=40 --visualize \
#    --infer_img=demo/000000014439_640x640.jpg \
#    -o FasterRCNNTestFeed.batch_size=1
## Fluid FP32
#python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/$CONFIG --mode=fluid --visualize \
#    --infer_img=demo/000000014439_640x640.jpg \
#    -o FasterRCNNTestFeed.batch_size=4
    #--infer_img=demo/000000014439_640x640.jpg \

#python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/$CONFIG --mode=fluid --is_eval


# MaskRCNN 
CONFIG=mask_rcnn_r50_1x
MODLE=https://paddlemodels.bj.bcebos.com/object_detection/${CONFIG}.tar
# Save model
python tools/infer.py -c configs/${CONFIG}.yml --infer_img demo/000000014439.jpg -o weights=${MODLE} MaskRCNNTestFeed.image_shape=[3,640,640] --save_inference_model
# TensorRT FP32
#python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/$CONFIG --mode=trt_fp32 --min_subgraph_size=40 --visualize \
#    --infer_img=demo/000000014439_640x640.jpg \
#    -o FasterRCNNTestFeed.batch_size=1
## Fluid FP32
python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/$CONFIG --mode=fluid --min_subgraph_size=40 --visualize \
    --infer_img=demo/000000014439_640x640.jpg \
    -o MaskRCNNTestFeed.batch_size=1
    #--infer_img=demo/000000014439_640x640.jpg \


# RetinaNet 
CONFIG=retinanet_r50_fpn_1x
MODLE=https://paddlemodels.bj.bcebos.com/object_detection/${CONFIG}.tar
# Save model
#python tools/infer.py -c configs/${CONFIG}.yml --infer_img demo/000000014439.jpg -o weights=${MODLE} FasterRCNNTestFeed.image_shape=[3,640,640] --save_inference_model
## TensorRT FP32
#python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/$CONFIG --mode=trt_fp32 --visualize \
#    --infer_img=demo/000000014439_640x640.jpg \
#    -o FasterRCNNTestFeed.batch_size=1
#python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/$CONFIG --mode=trt_fp16 --visualize \
#    --infer_img=demo/000000014439_640x640.jpg \
#    -o FasterRCNNTestFeed.batch_size=1
### Fluid FP32
#python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/$CONFIG --mode=fluid --visualize \
#    --infer_img=demo/000000014439_640x640.jpg \
#    -o FasterRCNNTestFeed.batch_size=1


# Cascade
CONFIG=cascade_rcnn_r50_fpn_1x
MODLE=https://paddlemodels.bj.bcebos.com/object_detection/${CONFIG}.tar
# Save model
#python tools/infer.py -c configs/${CONFIG}.yml --infer_img demo/000000014439.jpg -o weights=${MODLE} FasterRCNNTestFeed.image_shape=[3,640,640] --save_inference_model
# TensorRT FP32
#python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/$CONFIG --mode=trt_fp32 --min_subgraph_size=50 --visualize \
#    --infer_img=demo/000000014439_640x640.jpg \
#    -o FasterRCNNTestFeed.batch_size=1
## TensorRT FP16
#python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/$CONFIG --mode=trt_fp16 --min_subgraph_size=40 --visualize \
#    --infer_img=demo/000000014439_640x640.jpg \
#    -o FasterRCNNTestFeed.batch_size=1
## Fluid FP32
#python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/$CONFIG --mode=fluid --visualize \
#    --infer_img=demo/000000014439_640x640.jpg \
#    -o FasterRCNNTestFeed.batch_size=1

CONFIG=cascade_rcnn_dcn_r101_vd_fpn_1x
MODLE=https://paddlemodels.bj.bcebos.com/object_detection/${CONFIG}.tar
## Save model
#python tools/infer.py -c configs/dcn/${CONFIG}.yml --infer_img demo/000000014439.jpg -o weights=${MODLE} FasterRCNNTestFeed.image_shape=[3,640,640] --save_inference_model
# TensorRT FP32
#python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/$CONFIG --mode=trt_fp32 --min_subgraph_size=50 --visualize \
#    --infer_img=demo/000000014439_640x640.jpg \
#    -o FasterRCNNTestFeed.batch_size=1
## TensorRT FP16
#python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/$CONFIG --mode=trt_fp16 --min_subgraph_size=40 --visualize \
#    --infer_img=demo/000000014439_640x640.jpg \
#    -o FasterRCNNTestFeed.batch_size=1
## Fluid FP32
#python tools/tensorrt.py -c configs/dcn/${CONFIG}.yml --model_path=output/$CONFIG --mode=fluid --visualize \
#    --infer_img=demo/000000014439_640x640.jpg \
#    -o FasterRCNNTestFeed.batch_size=1
