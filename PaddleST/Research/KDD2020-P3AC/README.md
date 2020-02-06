# P3AC

## 任务说明(Introduction)

TODO

## 安装说明(Install Guide)

### 环境准备

1. paddle安装

    本项目依赖于Paddle Fluid 1.6.1 及以上版本，请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装

2. 下载代码

    克隆数据集代码库到本地, 本代码依赖[Paddle-EPEP框架](https://github.com/PaddlePaddle/epep)
    ```
    git clone https://github.com/PaddlePaddle/epep.git
    cd epep
    git clone https://github.com/PaddlePaddle/models.git
    ln -s models/PaddleST/Research/KDD2020-P3AC/conf/poi_qac_personalized conf/poi_qac_personalized
    ln -s models/PaddleST/Research/KDD2020-P3AC/datasets/poi_qac_personalized datasets/poi_qac_personalized
    ln -s models/PaddleST/Research/KDD2020-P3AC/nets/poi_qac_personalized nets/poi_qac_personalized
    ```

3. 环境依赖

    python版本依赖python 2.7


### 实验说明

1. 数据准备

    TODO
    ```
    #script to download 
    ```

2. 模型训练

    ```
    cp conf/poi_qac_personalized/poi_qac_personalized.local.conf.template conf/poi_qac_personalized/poi_qac_personalized.local.conf
    sh run.sh -c conf/poi_qac_personalized/poi_qac_personalized.local.conf -m train [ -g 0 ]
    ```

3. 模型评估
    ```
    pred_gpu=$1
    mode=$2 #query, poi, eval

    if [ $# -lt 2 ];then
        exit 1
    fi

    #编辑conf/poi_qac_personalized/poi_qac_personalized.local.conf.template，打开 CUDA_VISIBLE_DEVICES: <pred_gpu>
    cp conf/poi_qac_personalized/poi_qac_personalized.local.conf.template conf/poi_qac_personalized/poi_qac_personalized.local.conf
    sed -i "s#<pred_gpu>#$pred_gpu#g" conf/poi_qac_personalized/poi_qac_personalized.local.conf
    sed -i "s#<mode>#$mode#g" conf/poi_qac_personalized/poi_qac_personalized.local.conf

    sh run.sh -c poi_qac_personalized.local -m predict 1>../tmp/$mode-pred$pred_gpu.out 2>../tmp/$mode-pred$pred_gpu.err
    ```

## 论文下载(Paper Download)

Please feel free to review our paper :)

TODO

## 引用格式(Paper Citation)

TODO


