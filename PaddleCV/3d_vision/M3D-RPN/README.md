# M3D-RPN: Monocular 3D Region Proposal Network for Object Detection



## Introduction


Monocular 3D region proposal network for object detection accepted to ICCV 2019 (Oral), detailed in [arXiv report](https://arxiv.org/abs/1907.06038).




## Setup

- **Cuda & Python**

    In this project we utilize PaddlePaddle1.8 with Python 3, Cuda 9, and a few Anaconda packages.

- **Data**

    Download the full [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) detection dataset. Then place a softlink (or the actual data) in  *M3D-RPN/data/kitti*.

    ```
    cd M3D-RPN
    ln -s /path/to/kitti dataset/kitti
    ```

    Then use the following scripts to extract the data splits, which use softlinks to the above directory for efficient storage.

    ```
    python dataset/kitti_split1/setup_split.py
    python dataset/kitti_split2/setup_split.py
    ```

    Next, build the KITTI devkit eval for each split.

    ```
    sh dataset/kitti_split1/devkit/cpp/build.sh
    sh dataset/kitti_split2/devkit/cpp/build.sh
    ```

    Lastly, build the nms modules

    ```
    cd lib/nms
    make
    ```

## Training


Training is split into a warmup and main configurations. Review the configurations in *config* for details.

```
// First train the warmup (without depth-aware)
python train.py --config=kitti_3d_multi_warmup

// Then train the main experiment (with depth-aware)
python train.py --config=kitti_3d_multi_main
```



## Testing

We provide models for the main experiments on val1 data splits available to download here [M3D-RPN-release.tar](https://pan.baidu.com/s/1VQa5hGzIbauLOQi-0kR9Hg), passward:ls39.

Testing requires paths to the configuration file and model weights, exposed variables near the top *test.py*. To test a configuration and model, simply update the variables and run the test file as below.

```
python test.py --conf_path M3D-RPN-release/conf.pkl --weights_path M3D-RPN-release/iter50000.0_params.pdparams
```
