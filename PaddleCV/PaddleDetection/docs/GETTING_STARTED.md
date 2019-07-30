# Getting Started

For setting up the running environment, please refer to [installation
instructions](INSTALL.md).


## Training


#### Single-GPU Training


```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/faster_rcnn_r50_1x.yml
```

#### Multi-GPU Training


```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tools/train.py -c configs/faster_rcnn_r50_1x.yml
```

- Datasets is stored in `dataset/coco` by default (configurable).
- Datasets will be downloaded automatically and cached in `~/.cache/paddle/dataset` if not be found locally.
- Pretrained model is downloaded automatically and cached in `~/.cache/paddle/weights`.
- Model checkpoints is saved in `output` by default (configurable).
- To check out hyper parameters used, please refer to the config file.
- RCNN models training on CPU is not supported on PaddlePaddle<=1.5.1 and will be fixed on later version.


Alternating between training epoch and evaluation run is possible, simply pass
in `--eval` to do so and evaluate at each snapshot_iter. If evaluation dataset is large and
causes time-consuming in training, we suggest decreasing evaluation times or evaluating after training.

## Evaluation


```bash
export CUDA_VISIBLE_DEVICES=0
# or run on CPU with:
# export CPU_NUM=1
python tools/eval.py -c configs/faster_rcnn_r50_1x.yml
```

- Checkpoint is loaded from `output` by default (configurable)
- Multi-GPU evaluation for R-CNN and SSD models is not supported at the
moment, but it is a planned feature


## Inference


- Run inference on a single image:

```bash
export CUDA_VISIBLE_DEVICES=0
# or run on CPU with:
# export CPU_NUM=1
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml --infer_img=demo/000000570688.jpg
```

- Batch inference:

```bash
export CUDA_VISIBLE_DEVICES=0
# or run on CPU with:
# export CPU_NUM=1
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml --infer_dir=demo
```

The visualization files are saved in `output` by default, to specify a different
path, simply add a `--output_dir=` flag.

- Save inference model

```bash
export CUDA_VISIBLE_DEVICES=0
# or run on CPU with:
# export CPU_NUM=1
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml --infer_img=demo/000000570688.jpg \
                      --save_inference_model
```

Save inference model by set `--save_inference_model`, which can be loaded by PaddlePaddle predict library.


## FAQ

**Q:**  Why do I get `NaN` loss values during single GPU training? </br>
**A:**  The default learning rate is tuned to multi-GPU training (8x GPUs), it must
be adapted for single GPU training accordingly (e.g., divide by 8).


**Q:**  How to reduce GPU memory usage? </br>
**A:**  Setting environment variable FLAGS_conv_workspace_size_limit to a smaller
number can reduce GPU memory footprint without affecting training speed.
Take Mask-RCNN (R50) as example, by setting `export FLAGS_conv_workspace_size_limit=512`,
batch size could reach 4 per GPU (Tesla V100 16GB).
