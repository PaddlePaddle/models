# Getting Started

Please refer [installation instructions](INSTALL.md) to install PaddlePaddle and prepare dataset at first.


## Train a Model

#### One-Device Training

```bash
export CUDA_VISIBLE_DEVICES=0
# export CPU_NUM=1 # for CPU training
python tools/train.py -c configs/faster_rcnn_r50_1x.yml
```

#### Multi-Device Training

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # set devices
# export CPU_NUM=8 # for CPU training
python tools/train.py -c =configs/faster_rcnn_r50_1x.yml
```

- Default dataset directory is `dataset/coco`, users also can specify it in the configure file.
- Pretrained model will be downloaded automatically.
- Model will be saved at  `output/faster_rcnn_r50_1x` by default, users also can specify it in the configure file.
- All hyper parameters can refer input config.
- Change config file for other models.
- For more help, please run `python tools/train.py --help`.


For `SSD` on Pascal-VOC dataset,  set `--eval=True` to do evaluation during training.
For other models based on COCO dataset, the evaluating during training is not fully verified, better to do evaluation after traning.


### Distributed Training

Will add distributed training guide later.


## Evaluate with Pretrained models.


```bash
export CUDA_VISIBLE_DEVICES=0
# export CPU_NUM=1 # for CPU training
python tools/eval.py -c configs/faster_rcnn_r50_1x.yml
```

- The default model directory is `output/faster_rcnn_r50_1x`, you also can specify it.
- For R-CNN and SSD models, do not support evaluating by multi-device now, we will enhanced it in next version.
- For more help, please run `python tools/eval.py --help`.


## Inference with Pretrained Models

- Infer one image:

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml --infer_img=demo/000000000139.jpg
```

- Infer several images:

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml --infer_dir=demo
```

The predicted and visualized images are saved in `output` by default, uers also can change the saved directory by specifying `--savefile=`.  For more help please run `python tools/infer.py --helo`.


## FAQ


Q: Why the loss may be NaN when using single GPU to train?
A: The default learning rate is adapt to multi-device training, when use single GPU and small batch size, you need to decrease `base_lr` by corresponding multiples.  
