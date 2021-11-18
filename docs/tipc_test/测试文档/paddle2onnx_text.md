## 背景

Paddle2ONNX 预测功能测试的主程序为 `test_paddle2onnx.sh`，可以测试基于 Paddle2ONNX 的部署功能。本文介绍 Paddle2ONNX 预测功能测试文档的撰写规范。

## 文档规范

本文档和[基础训练预测文档](todo:add_basic_link)大体结构类似，主要去掉了训练相关的部分，文档目录结构如下：

### 1.测试结论汇总

内容：基于训练是否使用量化，可以将训练产出的模型可以分为`正常模型`和`量化模型`，表格形式给出这两类模型对应的预测功能汇总情况，包含`模型类型`、`device`这2个字段。

- device：CPU 或 GPU，其他字段取值对应本字段指定的环境；

示例：

| 模型类型 |device |
|  ----   |  ---- |  
| 正常模型 | GPU |
| 正常模型 | CPU |
| 量化模型 | GPU |
| 量化模型 | CPU |


### 2.测试流程
#### 2.1 功能测试
内容：给出 Paddle2ONNX 预测具体测试命令。
示例：

先运行`prepare.sh`准备数据和模型，然后运行`test_paddle2onnx.sh`进行测试，最终在```test_tipc/output```目录下生成`paddle2onnx_infer_*.log`后缀的日志文件。

```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ppocr_det_mobile/model_linux_gpu_normal_normal_paddle2onnx_python_linux_cpu.txt "paddle2onnx_infer"

# 用法:
bash test_tipc/test_paddle2onnx.sh ./test_tipc/configs/ppocr_det_mobile/model_linux_gpu_normal_normal_paddle2onnx_python_linux_cpu.txt
```  

#### 运行结果

各测试的运行情况会打印在 `test_tipc/output/results_paddle2onnx.log` 中：
运行成功时会输出：

```
Run successfully with command -  paddle2onnx  --model_dir=./inference/ch_ppocr_mobile_v2.0_det_infer/ --model_filename=inference.pdmodel --params_filename=inference.pdiparams --save_file=./inference/det_mobile_onnx/model.onnx --opset_version=10  --enable_onnx_checker=True!
Run successfully with command - python test_tipc/onnx_inference/predict_det.py --use_gpu=False --image_dir=./inference/ch_det_data_50/all-sum-510/ --det_model_dir=./inference/det_mobile_onnx/model.onnx  2>&1 !
```

运行失败时会输出：

```
Run failed with command - paddle2onnx  --model_dir=./inference/ch_ppocr_mobile_v2.0_det_infer/ --model_filename=inference.pdmodel --params_filename=inference.pdiparams --save_file=./inference/det_mobile_onnx/model.onnx --opset_version=10  --enable_onnx_checker=True!
...
```

#### 2.2 精度测试

由于 Paddle2ONNX 调用了 onnx 预测，因此不做精度对比。
