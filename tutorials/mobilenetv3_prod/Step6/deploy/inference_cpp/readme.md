# 服务器端C++预测

本教程将介绍在服务器端部署mobilenet_v3_small模型的详细步骤。


## 1. 准备环境

### 运行准备
- Linux环境，推荐使用docker。

### 1.1 编译opencv库

* 首先需要从opencv官网上下载在Linux环境下源码编译的包，以3.4.7版本为例，下载及解压缩命令如下：

```
wget https://github.com/opencv/opencv/archive/3.4.7.tar.gz
tar -xvf 3.4.7.tar.gz
```

最终可以在当前目录下看到`opencv-3.4.7/`的文件夹。

* 编译opencv，首先设置opencv源码路径(`root_path`)以及安装路径(`install_path`)，`root_path`为下载的opencv源码路径，`install_path`为opencv的安装路径。在本例中，源码路径即为当前目录下的`opencv-3.4.7/`。

```shell
cd ./opencv-3.4.7
export root_path=$PWD
export install_path=${root_path}/opencv3
```

* 然后在opencv源码路径下，按照下面的方式进行编译。

```shell
rm -rf build
mkdir build
cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX=${install_path} \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DWITH_IPP=OFF \
    -DBUILD_IPP_IW=OFF \
    -DWITH_LAPACK=OFF \
    -DWITH_EIGEN=OFF \
    -DCMAKE_INSTALL_LIBDIR=lib64 \
    -DWITH_ZLIB=ON \
    -DBUILD_ZLIB=ON \
    -DWITH_JPEG=ON \
    -DBUILD_JPEG=ON \
    -DWITH_PNG=ON \
    -DBUILD_PNG=ON \
    -DWITH_TIFF=ON \
    -DBUILD_TIFF=ON

make -j
make install
```

* `make install`完成之后，会在该文件夹下生成opencv头文件和库文件，用于后面的代码编译。

以opencv3.4.7版本为例，最终在安装路径下的文件结构如下所示。**注意**：不同的opencv版本，下述的文件结构可能不同。

```
opencv3/
|-- bin
|-- include
|-- lib64
|-- share
```

### 1.2 下载或者编译Paddle预测库

* 有2种方式获取Paddle预测库，下面进行详细介绍。

#### 1.2.1 预测库源码编译
* 如果希望获取最新预测库特性，可以从Paddle github上克隆最新代码，源码编译预测库。
* 可以参考[Paddle预测库官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#id16)的说明，从github上获取Paddle代码，然后进行编译，生成最新的预测库。使用git获取代码方法如下。

```shell
git clone https://github.com/PaddlePaddle/Paddle.git
```

* 进入Paddle目录后，使用如下方法编译。

```shell
rm -rf build
mkdir build
cd build

cmake  .. \
    -DWITH_CONTRIB=OFF \
    -DWITH_MKL=ON \
    -DWITH_MKLDNN=ON  \
    -DWITH_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_INFERENCE_API_TEST=OFF \
    -DON_INFER=ON \
    -DWITH_PYTHON=ON
make -j
make inference_lib_dist
```

更多编译参数选项可以参考Paddle C++预测库官网：[https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#id16](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#id16)。


* 编译完成之后，可以在`build/paddle_inference_install_dir/`文件下看到生成了以下文件及文件夹。

```
build/paddle_inference_install_dir/
|-- CMakeCache.txt
|-- paddle
|-- third_party
|-- version.txt
```

其中`paddle`就是之后进行C++预测时所需的Paddle库，`version.txt`中包含当前预测库的版本信息。

#### 1.2.2 直接下载安装

* [Paddle预测库官网](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html)上提供了不同cuda版本的Linux预测库，可以在官网查看并选择合适的预测库版本。

  以`manylinux_cuda11.1_cudnn8.1_avx_mkl_trt7_gcc8.2`版本为例，使用下述命令下载并解压：


```shell
wget https://paddle-inference-lib.bj.bcebos.com/2.2.2/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz

tar -xvf paddle_inference.tgz
```


最终会在当前的文件夹中生成`paddle_inference/`的子文件夹。


## 2 开始运行

### 2.1 将模型导出为inference model

* 可以参考[模型导出](../../tools/export_model.py)，导出`inference model`，用于模型预测。得到预测模型后，假设模型文件放在`inference`目录下，则目录结构如下。

```
mobilenet_v3_small_infer/
|--inference.pdmodel
|--inference.pdiparams
|--inference.pdiparams.info
```
**注意**：上述文件中，`inference.pdmodel`文件存储了模型结构信息，`inference.pdiparams`文件存储了模型参数信息。注意两个文件的路径需要与配置文件`tools/config.txt`中的`cls_model_path`和`cls_params_path`参数对应一致。

### 2.2 编译 C++预测demo

* 编译命令如下，其中Paddle C++预测库、opencv等其他依赖库的地址需要换成自己机器上的实际地址。


```shell
sh tools/build.sh
```

具体地，`tools/build.sh`中内容如下。

```shell
OPENCV_DIR=your_opencv_dir
LIB_DIR=your_paddle_inference_dir
CUDA_LIB_DIR=your_cuda_lib_dir
CUDNN_LIB_DIR=your_cudnn_lib_dir
TENSORRT_DIR=your_tensorrt_lib_dir

BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DPADDLE_LIB=${LIB_DIR} \
    -DWITH_MKL=ON \
    -DDEMO_NAME=clas_system \
    -DWITH_GPU=OFF \
    -DWITH_STATIC_LIB=OFF \
    -DWITH_TENSORRT=OFF \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \

make -j
```

上述命令中，

* `OPENCV_DIR`为opencv编译安装的地址（本例中为`opencv-3.4.7/opencv3`文件夹的路径）；

* `LIB_DIR`为下载的Paddle预测库（`paddle_inference`文件夹），或编译生成的Paddle预测库（`build/paddle_inference_install_dir`文件夹）的路径；

* `CUDA_LIB_DIR`为cuda库文件地址，在docker中为`/usr/local/cuda/lib64`；

* `CUDNN_LIB_DIR`为cudnn库文件地址，在docker中为`/usr/lib64`。

* `TENSORRT_DIR`是tensorrt库文件地址，在dokcer中为`/usr/local/TensorRT-7.2.3.4/`，TensorRT需要结合GPU使用。

在执行上述命令，编译完成之后，会在当前路径下生成`build`文件夹，其中生成一个名为`clas_system`的可执行文件。


### 运行demo
* 首先修改`tools/config.txt`中对应字段：
  * use_gpu：是否使用GPU；
  * gpu_id：使用的GPU卡号；
  * gpu_mem：显存；
  * cpu_math_library_num_threads：底层科学计算库所用线程的数量；
  * use_mkldnn：是否使用MKLDNN加速；
  * use_tensorrt: 是否使用tensorRT进行加速；
  * use_fp16：是否使用半精度浮点数进行计算，该选项仅在use_tensorrt为true时有效；
  * cls_model_path：预测模型结构文件路径；
  * cls_params_path：预测模型参数文件路径；
  * resize_short_size：预处理时图像缩放大小；
  * crop_size：预处理时图像裁剪后的大小。

* 然后修改`tools/run.sh`：
  * `./build/clas_system ./tools/config.txt /work/Docs/models/tutorials/mobilenetv3_prod/Step6/images/demo.jpg`
  * 上述命令中分别为：编译得到的可执行文件`clas_system`；运行时的配置文件`config.txt`；待预测的图像。

* 最后执行以下命令，完成对一幅图像的分类。

```shell
sh tools/run.sh
```
对于下面的图像进行预测

<div align="center">
    <img src="../../images/demo.jpg" width=300">
</div>

* 最终屏幕上会输出结果，如下图所示。
>pu_math_library_num_threads : 10
crop_size : 224
gpu_id : 0
gpu_mem : 4000
resize_short_size : 256
use_fp16 : 0
use_gpu : 0
use_mkldnn : 1
use_tensorrt : 0
=======End of Paddle Class inference config======
img_file_list length: 1
result:
	class id: 8
	score: 0.9014717937
Current image path: /work/Docs/models/tutorials/mobilenetv3_prod/Step6/images/demo.jpg
Current time cost: 0.0473620000 s, average time cost in all: 0.0473620000 s.

表示预测的类别ID是`8`，置信度为`0.901`，该结果与基于训练引擎的结果完全一致。
其中`class id`表示置信度最高的类别对应的id，score表示图片属于该类别的概率。
