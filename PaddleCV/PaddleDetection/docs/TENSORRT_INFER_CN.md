基于TensorRT的测试说明文档。

### 测试环境

- Python 2.7.1
- PaddlePaddle > 1.5
- CUDA 9.0
- CUDNN 7.5
- TensorRT 5.1


### 基于Develop编译TensorRT版本Paddle

可如下方式设置TensorRT的LIB，编译后安装whl包即可。

```
TRT=/paddle/TensorRT-5.1.5.0/
cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCUDNN_ROOT=/paddle/work/trt/cudnnv7.5_cuda9.0/ \
      -DCMAKE_INSTALL_PREFIX=`pwd`/output \
      -DWITH_PYTHON=ON   \
      -DON_INFER=ON \
      -DCUDA_ARCH_NAME=Auto \
      -DTENSORRT_INCLUDE_DIR=/paddle/work/trt/TensorRT-5.1.5.0/include \
      -DTENSORRT_LIBRARY=/paddle/work/trt/TensorRT-5.1.5.0/lib \

make -j20
make install
```

### 拉取分支

在TRT测试代码没有合入时 (暂时可能考虑不合入)，需要下面方式拉取代码。

```
git remote add qingqing01 https://github.com/qingqing01/models.git
git fetech qingqing01
git branch trt_infer qingqing01/trt_infer
git checkout trt_infer
```


### 测试流程

测试入口脚本是`tools/tensorrt.py`，先对该脚本说明，包含功能如何如下:

1. 支持基于Fluid测试，设置`mode=fluid`。
2. 支持基于Fluid-TensorRT测试，设置`mode=trt_fp32`，需要注意，TensorRT只支持输入定长图片，序列化的网络结构中的图片CHW，必需和输入真实数据的大小一致。
3. 支持基于Fluid-TensorRT INT8测试，但需要先运行一次通过calibiration生成校验表，再次运行测试速度。通过设置`mode=trt_int8`。
4. 支持对预测结果可视化，设置`--visualize`。
5. 支持使用C++预测引擎(Fluid or Fluid-TensorRT)进行预测、并评估mAP值，设置`--is_eval`。
6. 支持batch_size预测，通过设置EvalFeed中的batch_size，可通过命令参数`-o`设置，下面会具体给出例子。


预测速度测试流程包含2步:

1. 通过`tools/infer.py`保存模型和网路结构。
2. 通过`tools/tensorrt.py`测试不同模式的速度。


下面以SSD、Yolov3、FasterRCNN为例，给出预测速度测试命令:

- MobileNet-SSD:

```
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=`pwd`:$PYTHONPATH

rm -rf output/000000014439.jpg

# SSD on VOC
CONFIG=ssd_mobilenet_v1_voc
# 1. save model
python tools/infer.py -c configs/${CONFIG}.yml --infer_img demo/000000014439.jpg -o weights=https://paddlemodels.bj.bcebos.com/object_detection/${CONFIG}.tar --save_inference_model
# 2. Fluid
python tools/tensorrt.py -c configs/ssd/${CONFIG}.yml --model_path=output/${CONFIG} -o SSDEvalFeed.batch_size=16 --mode=fluid --visualize
# 3. TensorRT
python tools/tensorrt.py -c configs/ssd/${CONFIG}.yml --model_path=output/${CONFIG} -o SSDEvalFeed.batch_size=16 --mode=trt_fp32 --visualize

```


- Yolov3 on COCO:

网络默认的shape是3x608x608，这里通过命令参数改变输入的shape，如需改变输入大小，需要下面命令中的image_shape，target_size，比如下面都设置320。

```
# Yolo on COCO
CONFIG=yolov3_mobilenet_v1 # yolov3_darknet or yolov3_mobilenet_v1 or yolov3_r34
MODEL=https://paddlemodels.bj.bcebos.com/object_detection/${CONFIG}.tar
TRANSFORM="[!DecodeImage {to_rgb: true},!ResizeImage {target_size: 320,interp: 2},!NormalizeImage {mean: [0.485,0.456,0.406],std:[0.229,0.224,0.225],is_scale: true, is_channel_first: false},!Permute {to_bgr: false}]"
# 1. Save model by 320 x 320
python tools/infer.py -c configs/${CONFIG}.yml --infer_img demo/000000014439.jpg --save_inference_model \
    -o weights=${MODEL} YoloEvalFeed.image_shape=[3,320,320] YoloEvalFeed.sample_transforms="${TRANSFORM}"
# 2. Fluid FP32
python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/${CONFIG} --mode=fluid --visualize \
    -o YoloEvalFeed.batch_size=1 YoloEvalFeed.image_shape=[3,320,320] YoloEvalFeed.sample_transforms="$TRANSFORM" # SSDEvalFeed.samples=1 #可设置采用第一张图片
# 3. TensoRT FP32
python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/${CONFIG} --mode=trt_fp32 --visualize \
    -o YoloEvalFeed.batch_size=1 YoloEvalFeed.image_shape=[3,320,320] YoloEvalFeed.sample_transforms="${TRANSFORM}" # SSDEvalFeed.samples=1 #可设置采用第一张图片
```

- FasterRCNN on COCO

由于FasterRCNN的数据预处理逻辑是: 先使得短边为target_size(比如800)，等比例缩放，但当长变超过max_size (比如1333)时，则按照长边为max_size等比例缩放。每张图片大小不固定，下面示例是测试固定图片800x1333的速度。


```
# FasterRCNN
CONFIG=faster_rcnn_r50_1x
MODLE=https://paddlemodels.bj.bcebos.com/object_detection/${CONFIG}.tar
# Save model
python tools/infer.py -c configs/${CONFIG}.yml --infer_img demo/000000014439.jpg -o weights=${MODLE} --save_inference_model
# TensorRT FP32
python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/$CONFIG --mode=trt_fp32 --visualize \
    --infer_img=demo/000000014439_800x1333.jpg \
    -o FasterRCNNEvalFeed.batch_size=1
# Fluid FP32
python tools/tensorrt.py -c configs/${CONFIG}.yml --model_path=output/$CONFIG --mode=fluid --visualize \
    --infer_img=demo/000000014439_800x1333.jpg \
    -o FasterRCNNEvalFeed.batch_size=1
```
