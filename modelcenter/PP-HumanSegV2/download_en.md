# Model List

## 1 Portrait Segmentation Models

We release self-developed portrait segmentation models for real-time applications such as mobile video and web conferences. These models can be directly integrated into products at zero cost.

| Model Name | Best Input Shape | mIou(%) | Inference Time on Arm CPU(ms) | Modle Size(MB) | Config File | Links |
| --- | --- | --- | ---| --- | --- | --- |
| PP-HumanSegV1-Lite | 398x224 | 93.60 | 29.68 | 2.3 | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/portrait_pp_humansegv1_lite.yml) | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv1_lite_398x224_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv1_lite_398x224_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv1_lite_398x224_inference_model_with_softmax.zip) |
| PP-HumanSegV2-Lite | 256x144 | 96.63 | 15.86 | 5.4 | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/portrait_pp_humansegv2_lite.yml) | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv2_lite_256x144_smaller/portrait_pp_humansegv2_lite_256x144_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv2_lite_256x144_smaller/portrait_pp_humansegv2_lite_256x144_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv2_lite_256x144_smaller/portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax.zip) |

<details><summary>Note:</summary>

* Test the segmentation accuracy (mIoU): We test the above models on [PP-HumanSeg-14K](./paper.md) dataset with the best input shape.
* Test the inference time: Use [PaddleLite](https://www.paddlepaddle.org.cn/lite), xiaomi9 (Snapdragon 855 CPU), single thread, the best input shape.
* For the best input shape, the ratio of height and width is 16:9, which is the same as the camera of mobile phone and laptop.
* The checkpoint is the pretrained weight, which is used for finetune.
* Inference model is used for deployment.
* Inference Model (Argmax): The last operation of inference model is argmax, so the output has single channel.
* Inference Model (Softmax): The last operation of inerence model is softmax, so the output has two channels.
</details>

<details><summary>Usage:</summary>

* Portrait segmentation model can be directly integrated into products at zero cost.
* For mobile phone, there are horizontal and vertical screen. We need to rotate the image to keep the human direction always be vertical.
</details>

## 2 General Human Segmentation Models

For general human segmentation task, we first build a big human segmentation dataset, then use the SOTA model in PaddleSeg for training, finally release several general human segmentation models.


| Model Name | Best Input Shape | mIou(%) | Inference Time on ARM CPU(ms) | Inference Time on Nvidia GPU(ms) | Config File | Links |
| ----- | ---------- | ---------- | -----------------| ----------------- | ------- | ------- |
| PP-HumanSegV1-Lite   | 192x192 | 86.02 | 12.3  | -    | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/human_pp_humansegv1_lite.yml)   | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_lite_192x192_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_lite_192x192_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_lite_192x192_inference_model_with_softmax.zip) |
| PP-HumanSegV2-Lite   | 192x192 | 92.52 | 15.3  | -    | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/human_pp_humansegv2_lite.yml)   | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_lite_192x192_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_lite_192x192_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_lite_192x192_inference_model_with_softmax.zip) |
| PP-HumanSegV1-Mobile | 192x192 | 91.64 |  -    | 2.83 | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/human_pp_humansegv1_mobile.yml) | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_mobile_192x192_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_mobile_192x192_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_mobile_192x192_inference_model_with_softmax.zip) |
| PP-HumanSegV2-Mobile | 192x192 | 93.13 |  -    | 2.67 | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/human_pp_humansegv2_mobile.yml) | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_mobile_192x192_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_mobile_192x192_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax.zip) |
| PP-HumanSegV1-Server | 512x512 | 96.47 |  -    | 24.9 | [cfg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/configs/human_pp_humansegv1_server.yml) | [Checkpoint](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_server_512x512_pretrained.zip) \| [Inference Model (Argmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_server_512x512_inference_model.zip) \| [Inference Model (Softmax)](https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_server_512x512_inference_model_with_softmax.zip) |


<details><summary>Note:</summary>

* Test the segmentation accuracy (mIoU): After training the models on big human segmentation dataset, we test these models on small Supervisely Person dataset ([url](https://paddleseg.bj.bcebos.com/humanseg/data/mini_supervisely.zip)).
* Test the inference time: Use [PaddleLite](https://www.paddlepaddle.org.cn/lite), xiaomi9 (Snapdragon 855 CPU), single thread, the best input shape.
* The checkpoint is the pretrained weight, which is used for finetune.
* Inference model is used for deployment.
* Inference Model (Argmax): The last operation of inference model is argmax, so the output has single channel.
* Inference Model (Softmax): The last operation of inerence model is softmax, so the output has two channels.
</details>

<details><summary>Usage:</summary>

* Since the image of general human segmentation is various, you should evaluate the release model according to the actual scene.
* If the segmentation accuracy is not satisfied, you should annotate images and finetune the model with pretrained weights.
</details>