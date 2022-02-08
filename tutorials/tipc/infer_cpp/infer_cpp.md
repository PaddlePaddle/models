# Linux GPU/CPU 模型推理开发文档

# 目录

- [1. 简介](#1)
- [2. 推理过程开发](#2)
    - [2.1 准备系统环境](#2.1)
    - [2.2 准备输入数据和推理模型](#2.2)
    - [2.3 准备推理所需代码](#2.3)
    - [2.4 编译得到可执行代码](#2.4)
    - [2.5 运行得到结果](#2.5)
- [3. FAQ](#3)


## 1. 简介

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT进行预测加速，从而实现更优的推理性能。
更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/inference_cn.html)。
本文档主要介绍飞桨模型在 Linux GPU/CPU 下基于预测引擎的推理过程开发。


## 2. 推理过程开发

基于Paddle Inference的推理过程可以分为5个步骤，如下图所示。
<div align="center">
    <img src="../images/infer_cpp.png" width="600">
</div>

### 2.1 准备系统环境
* 配置合适的编译和执行环境，其中包括编译器，cuda等一些基础库，建议安装docker环境，[参考链接](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html)。
* 配置相应的paddle infer推理库，有两种方式，具体可以参考[链接](../../mobilenetv3_prod/Step6/deploy/inference_cpp/README.md#12-%E4%B8%8B%E8%BD%BD%E6%88%96%E8%80%85%E7%BC%96%E8%AF%91paddle%E9%A2%84%E6%B5%8B%E5%BA%93)。
* 配置安装第三库，例如opencv等。

### 2.2 准备输入数据和推理模型

**数据**

从验证集或者测试集中抽出至少一张图像，用于后续推理过程验证。

**推理模型**

对于训练好的模型，我们可以通过这种[方式](../train_infer_python/infer_python.md#22-%E5%87%86%E5%A4%87%E6%8E%A8%E7%90%86%E6%A8%A1%E5%9E%8B)获取用于推理的静态图模型。

### 2.3 准备推理所需代码

基于预测引擎的推理过程包含4个步骤：初始化预测引擎、预处理、推理、后处理。

**初始化预测引擎**

针对mobilenet_v3_small模型，推理引擎初始化函数实现如下，其中模型结构和参数文件路径、是否使用GPU、是否开启MKLDNN等内容都是可以配置的。
主要实现在[cls.cpp](../../mobilenetv3_prod/Step6/deploy/inference_cpp/src/cls.cpp)
```c++
void Classifier::LoadModel(const std::string &model_path,
                           const std::string &params_path) {
  paddle_infer::Config config;
  config.SetModel(model_path, params_path);

  if (this->use_gpu_) {
    config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
    if (this->use_tensorrt_) {
      config.EnableTensorRtEngine(
          1 << 20, 1, 3,
          this->use_fp16_ ? paddle_infer::Config::Precision::kHalf
                          : paddle_infer::Config::Precision::kFloat32,
          false, false);
    }
  } else {
    config.DisableGpu();
    if (this->use_mkldnn_) {
      config.EnableMKLDNN();
      // cache 10 different shapes for mkldnn to avoid memory leak
      config.SetMkldnnCacheCapacity(10);
    }
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
  }

  config.SwitchUseFeedFetchOps(false);
  // true for multiple input
  config.SwitchSpecifyInputNames(true);

  config.SwitchIrOptim(true);

  config.EnableMemoryOptim();
  config.DisableGlogInfo();

  this->predictor_ = CreatePredictor(config);
}

```
**预处理**

读取指定图像，对其进行数据变换，转化为符合模型推理所需要的输入格式。
* resize
* crop
* normalize
* RGB -> CHW
主要实现在[preprocess_op.cpp](../../mobilenetv3_prod/Step6/deploy/inference_cpp/src/preprocess_op.cpp)中。
```c++
//Resize
class ResizeImg {
public:
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img, int max_size_len);
};
//Crop
class CenterCropImg {
public:
  virtual void Run(cv::Mat &im, const int crop_size = 224);
};
//Norm
class Normalize {
public:
  virtual void Run(cv::Mat *im, const std::vector<float> &mean,
                   const std::vector<float> &scale, const bool is_scale = true);
};
// RGB -> CHW
class Permute {
public:
  virtual void Run(const cv::Mat *im, float *data);
};
```
**推理**

前向推理主要实现在[cls.cpp](../../mobilenetv3_prod/Step6/deploy/inference_cpp/src/cls.cpp)。
```C++
  auto input_names = this->predictor_->GetInputNames();
  auto input_t = this->predictor_->GetInputHandle(input_names[0]);
  input_t->Reshape({1, 3, resize_img.rows, resize_img.cols});
  auto start = std::chrono::system_clock::now();
  input_t->CopyFromCpu(input.data());
  this->predictor_->Run();

  std::vector<float> out_data;
  auto output_names = this->predictor_->GetOutputNames();
  auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());     
  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());
```

**后处理**

模型输出的是一个一维的数组，代表输入图片分类到每个类目的概率，为了得到有实际含义的输出，
需要获取该数组中最大值的位置和大小，mobilenet_v3_small的后处理代码如下所示。

```c++
int maxPosition = max_element(out_data.begin(), out_data.end()) - out_data.begin();
int score = out_data[maxPosition];
```

### 2.4 编译得到可执行代码
在准备好相应的代码后需要开始准备编译，这里可以利用cmake来实现，代码示例如：[CMakeLists.txt](../../mobilenetv3_prod/Step6/deploy/inference_cpp/CMakeLists.txt)
```bash
set(DEPS ${DEPS} ${OpenCV_LIBS})
AUX_SOURCE_DIRECTORY(./src SRCS)
add_executable(${DEMO_NAME} ${SRCS})
target_link_libraries(${DEMO_NAME} ${DEPS}) 
```
执行脚本：
```bash
OPENCV_DIR=../opencv-3.4.7/opencv3/
LIB_DIR=../paddle_inference/
CUDA_LIB_DIR=/usr/local/cuda/lib64
CUDNN_LIB_DIR=/usr/lib64
TENSORRT_DIR=/usr/local/TensorRT-7.2.3.4

BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DPADDLE_LIB=${LIB_DIR} \
    -DWITH_MKL=ON \
    -DWITH_GPU=OFF \
    -DWITH_STATIC_LIB=OFF \
    -DUSE_TENSORRT=OFF \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \

make -j
```

### 2.5 运行得到结果
相关脚本位置[run.sh](../../mobilenetv3_prod/Step6/deploy/inference_cpp/tools/run.sh)
```bash
./build/clas_system ./tools/config.txt ../../images/demo.jpg
```
## 3. FAQ
在上述配置中如果遇到相关问题可以参考[文档](https://paddleinference.paddlepaddle.org.cn/demo_tutorial/x86_linux_demo.html)以及[FAQ](https://paddleinference.paddlepaddle.org.cn/introduction/faq.html).