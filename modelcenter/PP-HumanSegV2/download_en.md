# Model List

## 1 Portrait Segmentation Models

Self-developed portrait segmentation models are released for real-time segmentation applications such as mobile phone video calls and web video conferences. These models can be directly integrated into products at zero cost.

| Model Name | Best Input Shape | mIou(%) | Inference Time on Phone(ms) | Modle Size(MB) | Config File | Links |
| --- | --- | --- | ---| --- | --- | --- |
| PP-HumanSegV1-Lite | 398x224 | 93.60 | 29.68 | 2.3 | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/portrait_pp_humansegv1_lite.yml) | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv1_lite_398x224_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv1_lite_398x224_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv1_lite_398x224_inference_model_with_softmax.zip) |
| PP-HumanSegV2-Lite | 256x144 | 96.63 | 15.86 | 5.4 | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/portrait_pp_humansegv2_lite.yml) | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv2_lite_256x144_smaller/portrait_pp_humansegv2_lite_256x144_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv2_lite_256x144_smaller/portrait_pp_humansegv2_lite_256x144_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv2_lite_256x144_smaller/portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax.zip) |

<details><summary>Note:</summary>

* Segmentation accuracy (mIoU): We tested the models on [PP-HumanSeg-14K](https://github.com/PaddlePaddle/models/pull/5583/paper.md) dataset using the best input shape, without tricks like multi-scale and flip.
* Inference latency on Snapdragon 855 ARM CPU: We tested the models on xiaomi9 (Snapdragon 855 CPU) using [PaddleLite](https://www.paddlepaddle.org.cn/lite), with single thread, large kernel and best input shape.
* For the best input shape, the ratio of width and height is 16:9, which is the same as the camera of mobile phone and laptop.
* The checkpoint is the pretrained weight, which can be used for finetune together with the config file.
* Inference model is used for deployment.
* Inference Model (Argmax): The last operation of inference model is argmax, so the output is a single-channel image (int64), where the portrait area is 1, and the background area is 0.
* Inference Model (Softmax): The last operation of inference model is softmax, so the output is a single-channel image(float32), where each pixel represents the probability of portrait area.
</details>

<details><summary>Usage:</summary>

* Portrait segmentation model can be directly integrated into products at zero cost. Best input shape is suggested.
* For mobile phone, screen can be horizontal or vertical. Before fed into model, the image should be rotated to keep the portrait always vertical.
</details>

## 2 General Human Segmentation Models

For general human segmentation task, we first built a large dataset, then used the SOTA models in PaddleSeg for training, and finally released several general models for human segmentation.


| Model Name | Best Input Shape | mIou(%) | Inference Time on ARM CPU(ms) | Inference Time on Nvidia GPU(ms) | Config File | Links |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- | ------- |
| PP-HumanSegV1-Lite   | 192x192 | 86.02 | 12.3  | -    | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/human_pp_humansegv1_lite.yml)   | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_lite_192x192_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_lite_192x192_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_lite_192x192_inference_model_with_softmax.zip) |
| PP-HumanSegV2-Lite   | 192x192 | 92.52 | 15.3  | -    | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/human_pp_humansegv2_lite.yml)   | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_lite_192x192_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_lite_192x192_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_lite_192x192_inference_model_with_softmax.zip) |
| PP-HumanSegV1-Mobile | 192x192 | 91.64 |  -    | 2.83 | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/human_pp_humansegv1_mobile.yml) | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_mobile_192x192_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_mobile_192x192_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_mobile_192x192_inference_model_with_softmax.zip) |
| PP-HumanSegV2-Mobile | 192x192 | 93.13 |  -    | 2.67 | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/human_pp_humansegv2_mobile.yml) | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_mobile_192x192_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_mobile_192x192_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax.zip) |
| PP-HumanSegV1-Server | 512x512 | 96.47 |  -    | 24.9 | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/human_pp_humansegv1_server.yml) | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_server_512x512_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_server_512x512_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_server_512x512_inference_model_with_softmax.zip) |


<details><summary>Note:</summary>

* Segmentation accuracy (mIoU): After training the models on large human segmentation dataset, we test these models on small Supervisely Person dataset ([url](https://paddleseg.bj.bcebos.com/humanseg/data/mini_supervisely.zip)).
* Inference latency on Snapdragon 855 ARM CPU: We tested the models on xiaomi9 (Snapdragon 855 CPU) using [PaddleLite](https://www.paddlepaddle.org.cn/lite), with single thread, large kernel and best input shape.
* Inference latency on server: We tested the models on Nvidia V100 GPU using Paddle Inference, with TensorRT and best input shape.
* The checkpoint is the pretrained weight, which can be used for finetune together with the config file.
* Inference model is used for deployment.
* Inference Model (Argmax): The last operation of inference model is argmax, so the output is a single-channel image (int64), where the portrait area is 1, and the background area is 0.
* Inference Model (Softmax): The last operation of inference model is softmax, so the output is a single-channel image(float32), where each pixel represents the probability of portrait area.
</details>

<details><summary>Usage:</summary>

* Since the images for general human segmentation are various, you should evaluate the released models according to the actual scene.
* If the segmentation accuracy is not satisfied, you are suggested to collect more data, annotate them and finetune the model.
</details>
