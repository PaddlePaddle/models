# MONOPOLY

## 简介

### 任务说明

互联网地图，作为一个经典的时空大数据平台，收集了大量有关固定资产（Point-of-Interest，简称POI)、出行轨迹、地点查询等相关信息。

Monopoly是一个POI商业智能算法，能够利用少量的房产价格，对大量其他的固定资产进行价值估计。

文章地址：XXXX

### 研究意义与发现

1）Monopoly能够帮助我们发现：各个城市居民对于不同类型公共资产价格评估的偏好，并且给出量化分析。

2）Monopoly能够帮助我们探索：不同城市居民对于私有房价评估的偏好，并且给出量化分析。

3）Monopoly能够帮助我们确定：评估一个固定资产价格需要考虑的空间范围。

### 效果说明


## 安装说明

1. paddle安装

    本项目依赖于Paddle Fluid 1.5.1 及以上版本，请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装

2. 下载代码

    克隆数据集代码库到本地, 本代码依赖[Paddle-EPEP框架](https://github.com/PaddlePaddle/epep)
    ```
    git clone https://github.com/PaddlePaddle/epep.git
    cd epep
    git clone https://github.com/PaddlePaddle/models.git
    ln -s models/PaddleST/Research/CIKM2019-MONOPOLY/conf/house_price conf/house_price
    ln -s models/PaddleST/Research/CIKM2019-MONOPOLY/datasets/house_price datasets/house_price
    ln -s models/PaddleST/Research/CIKM2019-MONOPOLY/nets/house_price nets/house_price
    ```

3. 环境依赖

    python版本依赖python 2.7


### 开始第一次模型调用
1. 数据准备

    TODO
    ```
    #script to download 
    ```

2. 模型训练

    ```
    sh run.sh -c conf/house_price/house_price.local.conf -m train [ -g 0 ]
    ```

3. 模型评估
    ```
    sh run.sh -c conf/house_price/house_price.local.conf -m pred
    cat $c.out | grep ^qid | python utils/calc_metric.py
    ```

## Reference this paper
=====


