# Installing PaddleDetection

---
## Table of Contents

- [Introduction](#introduction)
- [PaddlePaddle](#paddlepaddle)
- [Other Dependencies](#other-dependencies)
- [PaddleDetection](#paddle-detection)
- [Datasets](#datasets)
- [Update](#update)


## Introduction

This document covers how to install PaddleDetection, its dependencies (including PaddlePaddle), and the COCO and PascalVOC dataset.

For general information about PaddleDetection, please see [README.md](./README.md).


## PaddlePaddle

Running sample code in this directory requires PaddelPaddle Fluid v.1.5 and later. If the PaddlePaddle on your device is lower than this version, please follow the instructions in [installation document](http://www.paddlepaddle.org/documentation/docs/en/1.4/beginners_guide/install/index_en.html) and make an update.

Please make sure your PaddlePaddle installation was sucessful and the version of your PaddlePaddle is not lower than the version required. You can check PaddlePaddle installation with following commands.

```
# To check if PaddlePaddle installation was sucessful
python -c "from paddle.fluid import fluid; fluid.install_check.run_check()"

# To print PaddlePaddle version
python -c "import paddle; print(paddle.__version__)"
```


## Other Dependencies

**Install the [COCO-API](https://github.com/cocodataset/cocoapi):**

To train the model, COCO-API is needed. Installation is as follows:

    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    # if cython is not installed
    pip install Cython
    # Install into global site-packages
    make install
    # Alternatively, if you do not have permissions or prefer
    # not to install the COCO API into global site-packages
    python setup.py install --user


## PaddleDetection

**Clone Paddle models repository:**

You can clone Paddle models and change directory to PaddleDetection module with folloing commands:

```
cd <path/to/clone/models>
git clone https://github.com/PaddlePaddle/models
cd models/PaddleCV/object_detection
```

**Install python module requirements:**

Other python module requirements is set in [requirements.txt](./requirements.txt), you can install these requirements with folloing command:

```
pip install -r requirements.txt
```

**Check PaddleDetection architectures tests pass:**

```
python ppdet/modeling/tests/test_architectures.py
```


## Datasets

PaddleDetection support train/eval/infer models with dataset [MSCOCO](http://cocodataset.org) and [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/), you can set up dataset in following methods.

**Create symlinks for datasets:**

Dataset default path in PaddleDetection config files is `data/coco` and `data/voc`, you can set symlinks for your COCO/COCO-like or VOC/VOC-like datasets by following commands:

```
ln -sf <path/to/coco> $PaddleDetection/data/coco
ln -sf <path/to/voc> $PaddleDetection/data/voc
```

**Auto download datasets:**

If you set up models while `data/coc` and `data/voc` is not found, PaddleDetection will automaticly from [MSCOCO-2017](http://images.cocodataset.org) and [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC), decompress and save download datasets under `~/.cache/paddle/dataset/`. Once datasets has been download to `~/.cache/paddle/dataset/`, PaddleDetection will be able to find datasets in your next model settting up.


**NOTE:** For detail setting of datasets, please see [DATASET.md](./ppdet/data/README.md)

## Update

- 6/2019: Add installing PaddleDetection
