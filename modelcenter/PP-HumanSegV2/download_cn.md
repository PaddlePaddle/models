# 模型列表

## 1 肖像分割模型

针对手机视频通话、Web视频会议等实时半身人像的分割场景，PP-HumanSeg发布了自研的肖像分割模型。该系列模型可以开箱即用，零成本直接集成到产品中。


| 模型名 | 最佳输入尺寸 | 精度mIou(%) | 手机端推理耗时(ms) | 模型体积(MB) | 配置文件 | 下载连接 |
| --- | --- | --- | ---| --- | --- | --- |
| PP-HumanSegV1-Lite | 398x224 | 93.60 | 29.68 | 2.3 | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/portrait_pp_humansegv1_lite.yml) | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv1_lite_398x224_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv1_lite_398x224_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv1_lite_398x224_inference_model_with_softmax.zip) |
| PP-HumanSegV2-Lite | 256x144 | 96.63 | 15.86 | 5.4 | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/portrait_pp_humansegv2_lite.yml) | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv2_lite_256x144_smaller/portrait_pp_humansegv2_lite_256x144_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv2_lite_256x144_smaller/portrait_pp_humansegv2_lite_256x144_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv2_lite_256x144_smaller/portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax.zip) |

<details><summary>表格说明：</summary>

* 测试肖像模型的精度mIoU：针对PP-HumanSeg-14k数据集，使用模型最佳输入尺寸进行测试，没有应用多尺度和flip等操作。
* 测试肖像模型的推理耗时：基于[PaddleLite](https://www.paddlepaddle.org.cn/lite)预测库，小米9手机（骁龙855 CPU）、单线程、大核，使用模型最佳输入尺寸进行测试。
* 最佳输入尺寸的宽高比例是16:9，和手机、电脑的摄像头拍摄尺寸比例相同。
* Checkpoint是模型权重，结合模型配置文件，可以用于Finetuning场景。
* Inference Model为预测模型，可以直接用于部署。
* Inference Model (Argmax) 指模型最后使用Argmax算子，输出单通道预测结果(int64类型)，人像区域为1，背景区域为0。
* Inference Model (Softmax) 指模型最后使用Softmax算子，输出单通道预测结果（float32类型），每个像素数值表示是人像的概率。
</details>

<details><summary>使用说明：</summary>

* 肖像分割模型专用性较强，可以开箱即用，建议使用最佳输入尺寸。
* 在手机端部署肖像分割模型，存在横屏和竖屏两种情况。大家可以根据实际情况对图像进行旋转，保持人像始终是竖直，然后将图像（尺寸比如是256x144或144x256）输入模型，得到最佳分割效果。
</details>

## 2 通用人像分割模型

针对通用人像分割任务，我们首先构建的大规模人像数据集，然后使用PaddleSeg的SOTA模型，最终发布了多个PP-HumanSeg通用人像分割模型。


| 模型名 | 最佳输入尺寸 | 精度mIou(%) | 手机端推理耗时(ms) | 服务器端推理耗时(ms) | 配置文件 | 下载链接 |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- | ------- |
| PP-HumanSegV1-Lite   | 192x192 | 86.02 | 12.3  | -    | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/human_pp_humansegv1_lite.yml)   | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_lite_192x192_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_lite_192x192_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_lite_192x192_inference_model_with_softmax.zip) |
| PP-HumanSegV2-Lite   | 192x192 | 92.52 | 15.3  | -    | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/human_pp_humansegv2_lite.yml)   | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_lite_192x192_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_lite_192x192_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_lite_192x192_inference_model_with_softmax.zip) |
| PP-HumanSegV1-Mobile | 192x192 | 91.64 |  -    | 2.83 | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/human_pp_humansegv1_mobile.yml) | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_mobile_192x192_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_mobile_192x192_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_mobile_192x192_inference_model_with_softmax.zip) |
| PP-HumanSegV2-Mobile | 192x192 | 93.13 |  -    | 2.67 | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/human_pp_humansegv2_mobile.yml) | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_mobile_192x192_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_mobile_192x192_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax.zip) |
| PP-HumanSegV1-Server | 512x512 | 96.47 |  -    | 24.9 | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/human_pp_humansegv1_server.yml) | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_server_512x512_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_server_512x512_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_server_512x512_inference_model_with_softmax.zip) |


<details><summary>表格说明：</summary>

* 测试通用人像模型的精度mIoU：通用分割模型在大规模人像数据集上训练完后，在小规模Supervisely Person 数据集([下载链接](https://paddleseg.bj.bcebos.com/humanseg/data/mini_supervisely.zip))上进行测试。
* 测试手机端推理耗时：基于[PaddleLite](https://www.paddlepaddle.org.cn/lite)预测库，小米9手机（骁龙855 CPU）、单线程、大核，使用模型最佳输入尺寸进行测试。
* 测试服务器端推理耗时：基于[PaddleInference](https://www.paddlepaddle.org.cn/inference/product_introduction/inference_intro.html)预测裤，V100 GPU、开启TRT，使用模型最佳输入尺寸进行测试。
* Checkpoint是模型权重，结合模型配置文件，可以用于Finetune场景。
* Inference Model为预测模型，可以直接用于部署。
* Inference Model (Argmax) 指模型最后使用Argmax算子，输出单通道预测结果(int64类型)，人像区域为1，背景区域为0。
* Inference Model (Softmax) 指模型最后使用Softmax算子，输出单通道预测结果（float32类型），每个像素数值表示是人像的概率。
</details>

<details><summary>使用说明：</summary>

* 由于通用人像分割任务的场景变化很大，大家需要根据实际场景评估PP-HumanSeg通用人像分割模型的精度。
* 如果满足业务要求，可以直接应用到产品中。
* 如果不满足业务要求，大家可以收集、标注数据，基于开源通用人像分割模型进行Finetune。
</details>