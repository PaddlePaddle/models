# 飞桨视觉模型库PaddleCV

飞桨视觉模型库PaddleCV是基于飞桨的视觉统一预测部署和模型串联系统，覆盖图像分类，目标检测，图像分割，OCR等主流视觉方向。其中包括飞桨视觉模型库中的PP系列模型，例如PP-LCNet，PP-YOLOE，PP-OCR，PP-LiteSeg。用户可以通过安装包的方式快速进行推理，同时也支持灵活便捷的二次开发。

## <img src="https://user-images.githubusercontent.com/48054808/157827140-03ffaff7-7d14-48b4-9440-c38986ea378c.png" width="20"/> 覆盖模型

| 单模型/串联系统 | 方向 | 模型名称
|:------:|:-----:|:------:|
| 单模型 |   图像分类 |  [PP-LCNet](configs/single_op/PP-LCNet.yml) |
| 单模型 |   图像分类 |  [PP-LCNet v2](configs/single_op/PP-LCNetV2.yml) |
| 单模型 |   图像分类 |  [PP-HGNet](configs/single_op/PP-HGNet.yml) |
| 单模型 |   目标检测 |  [PP-YOLO](configs/single_op/PP-YOLO.yml) |
| 单模型 |   目标检测 |  [PP-YOLO v2](configs/single_op/PP-YOLOv2.yml) |
| 单模型 |   目标检测 |  [PP-YOLOE](configs/single_op/PP-YOLOE.yml) |
| 单模型 |   目标检测 |  [PP-YOLOE+](configs/single_op/PP-YOLOE+.yml) |
| 单模型 |   目标检测 |  [PP-PicoDet](configs/single_op/PP-PicoDet.yml) |
| 单模型 |   图像分割 |  [PP-HumanSeg v2](configs/single_op/PP-HumanSegV2.yml) |
| 单模型 |   图像分割 |  [PP-LiteSeg](configs/single_op/PP-LiteSeg.yml) |
| 单模型 |   图像分割 |  [PP-Matting v1](configs/single_op/PP-MattingV1.yml) |
| 串联系统 |   OCR |  [PP-OCR v2](configs/system/PP-OCRv2.yml) |
| 串联系统 |   OCR |  [PP-OCR v3](configs/system/PP-OCRv3.yml) |
| 串联系统 |   OCR |  [PP-Structure](configs/system/PP-Structure.yml) |
| 串联系统 |   图像识别 |  [PP-ShiTu](configs/system/PP-ShiTu.yml) |
| 串联系统 |   图像识别 |  [PP-ShiTu v2](configs/system/PP-ShiTuV2.yml) |
| 串联系统 |   行人分析 |  [PP-Human](configs/system/PP-Human.yml) |
| 串联系统 |   车辆分析 |  [PP-Vehicle](configs/system/PP-Vehicle.yml) |
| 串联系统 |   关键点检测 |  [PP-TinyPose](configs/system/PP-TinyPose.yml) |

## <img src="https://user-images.githubusercontent.com/48054808/157828296-d5eb0ccb-23ea-40f5-9957-29853d7d13a9.png" width="20"/> 文档教程

- [安装文档](docs/INSTALL.md)
- [使用教程](docs/GETTING_STARTED.md)
- [配置文件说明](docs/config_anno.md)
- 二次开发文档
  - [系统设计](docs/system_design.md)
  - [新增算子教程](docs/how_to_add_new_op.md)
  - [外部自定义算子](docs/custom_ops.md)

## <img title="" src="https://user-images.githubusercontent.com/48054808/157835345-f5d24128-abaf-4813-b793-d2e5bdc70e5a.png" alt="" width="20"> 许可证书

本项目的发布受[Apache 2.0 license](LICENSE)许可认证。
