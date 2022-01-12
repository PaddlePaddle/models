# Mobilenet_v3 在 ARM CPU 上部署示例

# 目录

- [1 获取 inference model]()
- [2 准备模型转换工具并生成 Paddle Lite 的部署模型]()
- [3 以 arm v8 、Android 系统为例进行部署]()
- [4 推理结果正确性验证]()


### 1 获取 inference model

提供以下两种方式获取 inference model 

- 直接下载：[inference model](https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_infer.tar)

- 通过预训练模型获取 

首先获取[预训练模型](https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_pretrained.pdparams)，在 ```models/tutorials/mobilenetv3_prod/Step6/tools``` 文件夹下提供了工具 export_model.py ，可以将预训练模型输出 为inference model ，运行如下命令即可获取 inference model。
```
# 假设当前在 models/tutorials/mobilenetv3_prod/Step6 目录下
python ./tools/export_model.py --pretrained=./mobilenet_v3_small_pretrained.pdparams  --save-inference-dir=./inference_model
```
在 inference_model 文件夹下有 inference.pdmodel、inference.pdiparams 和 inference.pdiparams.info 文件。

### 2 准备模型转换工具并生成 Paddle Lite 的部署模型

- 模型转换工具[opt_linux](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/opt_linux)、[opt_mac](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/opt_mac)。或者参考[文档](https://paddle-lite.readthedocs.io/zh/develop/user_guides/model_optimize_tool.html)编译您的模型转换工具

- 使用如下命令转换可以转换 inference model 到 Paddle Lite 的 nb 模型：

```
./opt --model_file=./inference_model/inference.pdmodel --param_file=./inference_model/inference.pdiparams --optimize_out=./mobilenet_v3_small
```
在当前文件夹下可以发现mobilenet_v3_small.nb文件。

注：在 mac 上运行 opt_mac 可能会有如下错误：

<div align="center">
    <img src="../../images/Paddle-Lite/pic1.png" width=400">
</div>
需要搜索安全性与隐私，点击通用，点击仍然允许，即可。
<div align="center">
    <img src="../../images/Paddle-Lite/pic2.png" width=500">
</div>

### 3 以 arm v8 、Android 系统为例进行部署

- 准备编译环境

```
gcc、g++（推荐版本为 8.2.0)   
git、make、wget、python、adb   
Java Environment   
CMake（请使用 3.10 版本,其他版本的 Cmake 可能有兼容性问题，导致编译不通过）
Android NDK（支持 ndk-r17c 及之后的所有 NDK 版本, 注意从 ndk-r18 开始，NDK 交叉编译工具仅支持 Clang, 不支持 GCC）  
```

- 环境安装命令

以 Ubuntu 为例介绍安装命令。注意需要 root 用户权限执行如下命令。mac 环境下编译 Android 库参考[Android 源码编译](https://paddle-lite.readthedocs.io/zh/develop/source_compile/macos_compile_android.html)，Windows 下暂不支持编译 Android 版本库。

```
   # 1. 安装 gcc g++ git make wget python unzip adb curl 等基础软件
   apt update
   apt-get install -y --no-install-recommends \
     gcc g++ git make wget python unzip adb curl

   # 2. 安装 jdk
   apt-get install -y default-jdk

   # 3. 安装 CMake，以下命令以 3.10.3 版本为例(其他版本的 Cmake 可能有兼容性问题，导致编译不通过，建议用这个版本)
   wget -c https://mms-res.cdn.bcebos.com/cmake-3.10.3-Linux-x86_64.tar.gz && \
       tar xzf cmake-3.10.3-Linux-x86_64.tar.gz && \
       mv cmake-3.10.3-Linux-x86_64 /opt/cmake-3.10 &&  
       ln -s /opt/cmake-3.10/bin/cmake /usr/bin/cmake && \
       ln -s /opt/cmake-3.10/bin/ccmake /usr/bin/ccmake

   # 4. 下载 linux-x86_64 版本的 Android NDK，以下命令以 r17c 版本为例，其他版本步骤类似。
   cd /tmp && curl -O https://dl.google.com/android/repository/android-ndk-r17c-linux-x86_64.zip
   cd /opt && unzip /tmp/android-ndk-r17c-linux-x86_64.zip

   # 5. 添加环境变量 NDK_ROOT 指向 Android NDK 的安装路径
   echo "export NDK_ROOT=/opt/android-ndk-r17c" >> ~/.bashrc
   source ~/.bashrc
```

- 获取预测库

可以使用下面两种方式获得预测库。

(1) 使用预编译包 

 推荐使用 Paddle Lite 仓库提供的 [release库](https://github.com/PaddlePaddle/Paddle-Lite/releases/tag/v2.10),在网页最下边选取要使用的库。

```
tar -xvzf inference_lite_lib.android.armv8.clang.c++_static.with_extra.with_cv.tar.gz
```
即可获取编译好的库。注意，即使获取编译好的库依然要进行上述**环境安装**的步骤，因为下面编译 demo 时候会用到。

(2) 编译预测库 

 运行编译脚本之前，请先检查系统环境变量 ``NDK_ROOT`` 指向正确的 Android NDK 安装路径。
之后可以下载并构建 Paddle Lite 编译包。

```
   # 1. 检查环境变量 `NDK_ROOT` 指向正确的 Android NDK 安装路径
   echo $NDK_ROOT

   # 2. 下载 Paddle Lite 源码并切换到发布分支，如 release/v2.10
   git clone https://github.com/PaddlePaddle/Paddle-Lite.git
   cd Paddle-Lite && git checkout release/v2.10

   # (可选) 删除 third-party 目录，编译脚本会自动从国内 CDN 下载第三方库文件
   # rm -rf third-party

   # 3. 编译 Paddle Lite Android 预测库
   ./lite/tools/build_android.sh
```

如果按 ``./lite/tools/build_android.sh`` 中的默认参数执行，成功后会在 ``Paddle-Lite/build.lite.android.armv8.gcc/inference_lite_lib.android.armv8`` 生成 Paddle Lite 编译包，文件目录如下。

```
   inference_lite_lib.android.armv8/
   ├── cxx                                               C++ 预测库和头文件
   │   ├── include                                       C++ 头文件
   │   │   ├── paddle_api.h
   │   │   ├── paddle_image_preprocess.h
   │   │   ├── paddle_lite_factory_helper.h
   │   │   ├── paddle_place.h
   │   │   ├── paddle_use_kernels.h
   │   │   ├── paddle_use_ops.h
   │   │   └── paddle_use_passes.h
   │   └── lib                                           C++ 预测库
   │       ├── libpaddle_api_light_bundled.a             C++ 静态库
   │       └── libpaddle_light_api_shared.so             C++ 动态库
   │
   ├── java                                              Java 预测库
   │   ├── jar
   │   │   └── PaddlePredictor.jar                       Java JAR 包
   │   ├── so
   │   │   └── libpaddle_lite_jni.so                     Java JNI 动态链接库
   │   └── src
   │
   └── demo                                              C++ 和 Java 示例代码
       ├── cxx                                           C++ 预测库示例
       └── java                                          Java 预测库示例
```

- 编译运行示例

将编译好的预测库放在当前目录下mobilenet_v3文件夹下，如下所示：

```
   mobilenet_v3/                                            示例文件夹
   ├── inference_lite_lib.android.armv8/                 Paddle Lite C++ 预测库和头文件
   │
   ├── Makefile                                          编译相关
   │
   ├── Makefile.def                                      编译相关
   │
   ├── mobilenet_v3_small.nb                             优化后的模型
   │
   ├── mobilenet_v3.cc                                   C++ 示例代码
   │
   ├── demo.jpg                                          示例图片
   │
   ├── imagenet1k_label_list.txt                         示例label(用于后处理)
   │
   └── config.txt                                        示例config(用于前处理)
```
在 mobilenet_v3 文件夹下运行

```bash
make
```
会进行编译过程，注意编译过程会下载 opencv 第三方库，需要连接网络。编译完成后会生成 mobilenet_v3可执行文件。
注意 Makefile 中第4行:

```
LITE_ROOT=./inference_lite_lib.android.armv8
```
中的 ```LITE_ROOT```需要改成您的预测库的文件夹名。

- 在 Android 手机上部署
连接一台开启了**USB调试功能**的手机，运行
```
adb devices
```
可以看到有输出
```
List of devices attached
1ddcf602	device
```

- 在手机上运行 mobilenet_v3 demo。

```bash
#################################
# 假设当前位于 mobilenet_v3 目录下   #
#################################

# prepare enviroment on phone
adb shell mkdir -p /data/local/tmp/arm_cpu/


# push executable binary, library to device
adb push mobilenet_v3 /data/local/tmp/arm_cpu/
adb shell chmod +x /data/local/tmp/arm_cpu/mobilenet_v3
adb push inference_lite_lib.android.armv8/cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/arm_cpu/

# push model with optimized(opt) to device
adb push ./mobilenet_v3_small.nb /data/local/tmp/arm_cpu/

# push config and label and pictures to device
adb push ./config.txt /data/local/tmp/arm_cpu/
adb push ./imagenet1k_label_list.txt /data/local/tmp/arm_cpu/
adb push ./demo.jpg /data/local/tmp/arm_cpu/

# run demo on device
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/arm_cpu/; \
           /data/local/tmp/arm_cpu/mobilenet_v3 \
           /data/local/tmp/arm_cpu/config.txt   \
           /data/local/tmp/arm_cpu/demo.jpg" 
```

得到以下输出：

```
===clas result for image: ./demo.jpg===
	Top-1, class_id: 494, class_name:  chime, bell, gong, score: 1
	Top-2, class_id: 0, class_name:  tench, Tinca tinca, score: 0
	Top-3, class_id: 0, class_name:  tench, Tinca tinca, score: 0
	Top-4, class_id: 0, class_name:  tench, Tinca tinca, score: 0
	Top-5, class_id: 0, class_name:  tench, Tinca tinca, score: 0

```

代表在 Android 手机上推理部署完成。

### 4 验证推理结果正确性

在`models/tutorials/mobilenetv3_prod/Step6`目录下运行如下命令：

```
python tools/predict.py --pretrained=./mobilenet_v3_small_paddle_pretrained.pdparams --img-path=images/demo.jpg
```
最终输出结果为 ```class_id: 8, prob: 0.9091238975524902``` ，表示预测的类别ID是```8```，置信度为```0.909```。

与Paddle Lite预测结果一致。
