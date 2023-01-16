## Model Zoo
### Keypoint Detection Model
| Model  | Input Size | AP (COCO Val) | Inference Time for Single Person (FP32)| Inference Time for Single Person（FP16) | Config | Model Weights | Deployment Model | Paddle-Lite Model（FP32) | Paddle-Lite Model（FP16)|
| :------------------------ | :-------:  | :------: | :------: |:---: | :---: | :---: | :---: | :---: | :---: |
| PP-TinyPose | 128*96 | 58.1 | 4.57ms | 3.27ms | [Config](./tinypose_128x96.yml) |[Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96.pdparams) | [Deployment Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96.tar) | [Lite Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96_lite.tar) | [Lite Model(FP16)](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96_fp16_lite.tar) |
| PP-TinyPose | 256*192 | 68.8 | 14.07ms | 8.33ms | [Config](./tinypose_256x192.yml) | [Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.pdparams) | [Deployment Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.tar) | [Lite Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192_lite.tar) | [Lite Model(FP16)](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192_fp16_lite.tar) |

### Pedestrian Detection Model
| Model  | Input Size | mAP (COCO Val) | Average Inference Time (FP32)| Average Inference Time (FP16) | Config | Model Weights | Deployment Model | Paddle-Lite Model（FP32) | Paddle-Lite Model（FP16)|
| :------------------------ | :-------:  | :------: | :------: | :---: | :---: | :---: | :---: | :---: | :---: |
| PicoDet-S-Pedestrian | 192*192 | 29.0 | 4.30ms |  2.37ms | [Config](../../picodet/legacy_model/application/pedestrian_detection/picodet_s_192_pedestrian.yml) |[Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_192_pedestrian.pdparams) | [Deployment Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_192_pedestrian.tar) | [Lite Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_192_pedestrian_lite.tar) | [Lite Model(FP16)](https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_192_pedestrian_fp16_lite.tar) |
| PicoDet-S-Pedestrian | 320*320 | 38.5 | 10.26ms |  6.30ms | [Config](../../picodet/legacy_model/application/pedestrian_detection/picodet_s_320_pedestrian.yml) | [Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_320_pedestrian.pdparams) | [Deployment Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_320_pedestrian.tar) | [Lite Model](https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_320_pedestrian_lite.tar) | [Lite Model(FP16)](https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_320_pedestrian_fp16_lite.tar) |


**Tips**
- The keypoint detection model and pedestrian detection model are both trained on `COCO train2017` and `AI Challenger trainset`. The keypoint detection model is evaluated on `COCO person keypoints val2017`, and the pedestrian detection model is evaluated on `COCO instances val2017`.
- The AP results of keypoint detection models are based on bounding boxes in GroundTruth.
- Both keypoint detection model and pedestrian detection model are trained in a 4-GPU environment. In practice, if number of GPUs or batch size need to be changed according to the training environment, you should refer to [FAQ](../../../docs/tutorials/FAQ/README.md) to adjust the learning rate.
- The inference time is tested on a Qualcomm Snapdragon 865, with 4 threads at arm8.
