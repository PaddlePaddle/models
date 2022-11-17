# 模型列表

## 1 Cityscapes上语义分割模型

| 模型名 | 骨干网络 | 训练迭代次数 | 训练输入尺寸 | 预测输入尺寸 | 精度mIoU | 精度mIoU(flip) | 精度mIoU(ms+flip) | 下载链接 |
| --- | --- | --- | ---| --- | --- | --- | --- | --- |
|PP-LiteSeg-T|STDC1|160000|1024x512|1025x512|73.10%|73.89%|-|[config](./pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml)\|[训练模型](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k/model.pdparams)\|[预测模型](https://paddleseg.bj.bcebos.com/inference/pp_liteseg_infer_models/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k_inference_model.zip)|
|PP-LiteSeg-T|STDC1|160000|1024x512|1536x768|76.03%|76.74%|-|[config](./pp_liteseg_stdc1_cityscapes_1024x512_scale0.75_160k.yml)\|[训练模型](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc1_cityscapes_1024x512_scale0.75_160k/model.pdparams)\|[预测模型](https://paddleseg.bj.bcebos.com/inference/pp_liteseg_infer_models/pp_liteseg_stdc1_cityscapes_1024x512_scale0.75_160k_inference_model.zip)|
|PP-LiteSeg-T|STDC1|160000|1024x512|2048x1024|77.04%|77.73%|77.46%|[config](./pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k.yml)\|[训练模型](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k/model.pdparams)\|[预测模型](https://paddleseg.bj.bcebos.com/inference/pp_liteseg_infer_models/pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k_inference_model.zip)|
|PP-LiteSeg-B|STDC2|160000|1024x512|1024x512|75.25%|75.65%|-|[config](./pp_liteseg_stdc2_cityscapes_1024x512_scale0.5_160k.yml)\|[训练模型](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc2_cityscapes_1024x512_scale0.5_160k/model.pdparams)\|[预测模型](https://paddleseg.bj.bcebos.com/inference/pp_liteseg_infer_models/pp_liteseg_stdc2_cityscapes_1024x512_scale0.5_160k_inference_model.zip)|
|PP-LiteSeg-B|STDC2|160000|1024x512|1536x768|78.75%|79.23%|-|[config](./pp_liteseg_stdc2_cityscapes_1024x512_scale0.75_160k.yml)\|[训练模型](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc2_cityscapes_1024x512_scale0.75_160k/model.pdparams)\|[预测模型](https://paddleseg.bj.bcebos.com/inference/pp_liteseg_infer_models/pp_liteseg_stdc2_cityscapes_1024x512_scale0.75_160k_inference_model.zip)|
|PP-LiteSeg-B|STDC2|160000|1024x512|2048x1024|79.04%|79.52%|79.85%|[config](./pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k.yml)\|[训练模型](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k/model.pdparams)\|[预测模型](https://paddleseg.bj.bcebos.com/inference/pp_liteseg_infer_models/pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k_inference_model.zip)|

## 2 CamVid上语义分割模型

| 模型名 | 骨干网络 | 训练迭代次数 | 训练输入尺寸 | 预测输入尺寸 | 精度mIoU | 精度mIoU(flip) | 精度mIoU(ms+flip) | 下载链接 |
| --- | --- | --- | ---| --- | --- | --- | --- | --- |
|PP-LiteSeg-T|STDC1|10000|960x720|960x720|73.30%|73.89%|73.66%|[config](./pp_liteseg_stdc1_camvid_960x720_10k.yml)\|[训练模型](https://paddleseg.bj.bcebos.com/dygraph/camvid/pp_liteseg_stdc1_camvid_960x720_10k/model.pdparams)\|[预测模型](https://paddleseg.bj.bcebos.com/inference/pp_liteseg_infer_models/pp_liteseg_stdc1_camvid_960x720_10k_inference_model.zip)|
|PP-LiteSeg-B|STDC2|10000|960x720|960x720|75.10%|75.85%|75.48%|[config](./pp_liteseg_stdc2_camvid_960x720_10k.yml)\|[训练模型](https://paddleseg.bj.bcebos.com/dygraph/camvid/pp_liteseg_stdc2_camvid_960x720_10k/model.pdparams)\|[预测模型](https://paddleseg.bj.bcebos.com/inference/pp_liteseg_infer_models/pp_liteseg_stdc2_camvid_960x720_10k_inference_model.zip)|
