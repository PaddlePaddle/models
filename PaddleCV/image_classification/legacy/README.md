For historical reasons, We keep "no name" models here, which are different from "specified name" models.

**NOTE: Training the models in legacy folder will generate models without specified parameters.**

- **Released models: not specify parameter names**

|model | top-1/top-5 accuracy(PIL)| top-1/top-5 accuracy(CV2) |
|- |:-: |:-:|
|[ResNet152](http://paddle-imagenet-models.bj.bcebos.com/ResNet152_pretrained.zip) | 78.18%/93.93% | 78.11%/94.04% |
|[SE_ResNeXt50_32x4d](http://paddle-imagenet-models.bj.bcebos.com/se_resnext_50_model.tar) | 78.32%/93.96% | 77.58%/93.73% |

---

2019/08/08
We move the dist_train and fp16 part to PaddlePaddle Fleet now.
and dist_train folder is temporary stored here.
