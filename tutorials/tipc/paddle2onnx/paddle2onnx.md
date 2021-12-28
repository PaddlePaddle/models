# Paddle2ONNX功能开发文档

# 目录

- [1. 简介](#1---)
- [2. Paddle2ONNX功能开发](#2---)
- [3. FAQ](#3---)

## 1. 简介
paddle2onnx支持将**PaddlePaddle**模型格式转化到**ONNX**模型格式。

- 模型格式，支持Paddle静态图和动态图模型转为ONNX，可转换由[save_inference_model](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/static/save_inference_model_cn.html#save-inference-model)导出的静态图模型，使用方法请参考[IPthon示例](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/examples/tutorial.ipynb)。动态图转换目前处于实验状态，将伴随Paddle 2.0正式版发布后，提供详细使用教程。
- 算子支持，目前稳定支持导出ONNX Opset 9~11，部分Paddle算子支持更低的ONNX Opset转换，详情可参考[算子列表](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/docs/zh/op_list.md)。
- 模型类型，官方测试可转换的模型请参考[模型库](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/docs/zh/model_zoo.md)。

## 2. Paddle2ONNX功能开发
### 2.1. 环境准备

需要准备 Paddle2ONNX 模型转化环境，和 ONNX 模型预测环境

####  Paddle2ONNX

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式，算子目前稳定支持导出 ONNX Opset 9~11，部分Paddle算子支持更低的ONNX Opset转换。
更多细节可参考 [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/README_zh.md)

- 安装 Paddle2ONNX
```
python3.7 -m pip install paddle2onnx
```

- 安装 ONNX
```
# 建议安装 1.9.0 版本，可根据环境更换版本号
python3.7 -m pip install onnxruntime==1.9.0
```

### 2.2. 模型转换


- Paddle 模型下载

有两种方式获取Paddle静态图模型：在 [model_list](../../doc/doc_ch/models_list.md) 中下载PaddleOCR提供的预测模型；
参考[模型导出说明](../../doc/doc_ch/inference.md#训练模型转inference模型)把训练好的权重转为 inference_model。

以 ppocr 检测模型为例：

```
wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar
cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && cd ..
```

- 模型转换

使用 Paddle2ONNX 将Paddle静态图模型转换为ONNX模型格式：

```
paddle2onnx --model_dir=./inference/ch_ppocr_mobile_v2.0_det_infer/ \
--model_filename=inference.pdmodel \
--params_filename=inference.pdiparams \
--save_file=./inference/det_mobile_onnx/model.onnx \
--opset_version=10 \
--enable_onnx_checker=True
```

执行完毕后，ONNX 模型会被保存在 `./inference/det_mobile_onnx/` 路径下

* 注意：以下几个模型暂不支持转换为 ONNX 模型：
NRTR、SAR、RARE、SRN

### 2.3. onnx 预测

以检测模型为例，使用 ONNX 预测可执行如下命令：

```
python3.7 ../../tools/infer/predict_det.py --use_gpu=False --use_onnx=True \
--det_model_dir=./inference/det_mobile_onnx/model.onnx \
--image_dir=../../doc/imgs/1.jpg
```

执行命令后在终端会打印出预测的检测框坐标，并在 `./inference_results/` 下保存可视化结果。

```
root INFO: 1.jpg  [[[291, 295], [334, 292], [348, 844], [305, 847]], [[344, 296], [379, 294], [387, 669], [353, 671]]]
The predict time of ../../doc/imgs/1.jpg: 0.06162881851196289
The visualized image saved in ./inference_results/det_res_1.jpg
```

* 注意：ONNX暂时不支持变长预测，需要将输入resize到固定输入，预测结果可能与直接使用Paddle预测有细微不同。

## 3. FAQ
