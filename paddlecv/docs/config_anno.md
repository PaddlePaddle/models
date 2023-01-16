# 配置文件说明

本文档以目标检测模型[PP-YOLOE+](../configs/single_op/PP-YOLOE+.yml)为例，具体说明配置文件各字段含义

## 环境配置段

```
ENV:
  min_subgraph_size: 3          # TensorRT最小子图大小
  trt_calib_mode: False         # 如果设置TensorRT离线量化校准，需要设置为True
  cpu_threads: 1                # CPU部署时线程数
  trt_use_static: False         # TensorRT部署是否加载预生成的engine文件
  save_img: True                # 是否保存可视化图片，默认路径在output文件夹下
  save_res: True                # 是否保存结构化输出，默认路径在output文件夹下
  return_res: True              # 是否返回全量结构化输出结果
```

## 模型配置段

```
MODEL:
  - DetectionOp:   # 模型算子类名，输出字段为固定值，即["dt_bboxes", "dt_scores", "dt_class_ids", "dt_cls_names"]
      name: det    # 算子名称，单个配置文件中不同算子名称不能重复
      param_path: paddlecv://models/ppyoloe_plus_crn_l_80e_coco/model.pdiparams    # 推理模型参数文件，支持本地地址，也支持线上链接并自动下载
      model_path: paddlecv://models/ppyoloe_plus_crn_l_80e_coco/model.pdmodel     # 推理模型文件，支持本地地址，也支持线上链接并自动下载
      batch_size: 1    # batch size
      image_shape: [3, *image_shape, *image_shape]    # 网络输入shape
      PreProcess:   # 预处理模块，集中在ppcv/ops/models/detection/preprocess.py中
        - Resize:
            interp: 2
            keep_ratio: false
            target_size: [*image_shape, *image_shape]
        - NormalizeImage:
            is_scale: true
            mean: [0., 0., 0.]
            std: [1., 1., 1.]
            norm_type: null
        - Permute:
      PostProcess:   #后处理模块，集中在ppcv/ops/models/detection/postprocess.py
        - ParserDetResults:
            label_list: paddlecv://dict/detection/coco_label_list.json
            threshold: 0.5
      Inputs:   # 输入字段，DetectionOp算子输入所需字段，格式为{上一个op名}.{上一个op输出字段}，第一个Op的上一个op名为input
        - input.image

  - DetOutput:    # 输出模型算子类名
      name: vis   # 算子名称，单个配置文件中不同算子名称不能重复
      Inputs:     # 输入字段，DetOutput算子输入所需字段，格式为{上一个op名}.{上一个op输出字段}
        - input.fn
        - input.image
        - det.dt_bboxes
        - det.dt_scores
        - det.dt_cls_names  
```
