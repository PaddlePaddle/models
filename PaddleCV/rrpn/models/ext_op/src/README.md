# 自定义OP的编译过程

## 代码结构

  - src: 扩展OP C++/CUDA 源码
  - rrpn_lib.py: Python封装

## 安装PaddlePaddle

请通过如下方式安装PaddlePaddle：

- 通过[Paddle develop分支](https://github.com/PaddlePaddle/Paddle/tree/develop)源码编译安装，编译方法如下:

  1. [Ubuntu](https://www.paddlepaddle.org.cn/install/doc/source/ubuntu)
  1. [CentOS](https://www.paddlepaddle.org.cn/install/doc/source/centos)
  1. [MasOS](https://www.paddlepaddle.org.cn/install/doc/source/macos)
  1. [Windows](https://www.paddlepaddle.org.cn/install/doc/source/windows)

  **说明：** 推荐使用docker编译

- 安装Paddle develop[每日版本whl包](https://www.paddlepaddle.org.cn/install/doc/tables#多版本whl包列表-dev-11)

  **注意：** 编译自定义OP使用的gcc版本须与Paddle编译使用gcc版本一致，Paddle develop每日版本目前采用**gcc 4.8.2**版本编译，若使用每日版本，请使用**gcc 4.8.2**版本编译自定义OP，否则可能出现兼容性问题。

## 编译自定义OP

自定义op需要将实现的C++、CUDA代码编译成动态库，mask.sh中通过g++/nvcc编译，当然您也可以写Makefile或者CMake。

编译需要include PaddlePaddle的相关头文件，链接PaddlePaddle的lib库。 头文件和lib库可通过下面命令获取到:

```
# python
>>> import paddle
>>> print(paddle.sysconfig.get_include())
/paddle/pyenv/local/lib/python2.7/site-packages/paddle/include
>>> print(paddle.sysconfig.get_lib())
/paddle/pyenv/local/lib/python2.7/site-packages/paddle/libs
```

我们提供动态库编译脚本如下：

```
cd src
sh make.sh
```

最终编译会产出`rrpn_lib.so`

**说明：** 若使用源码编译安装PaddlePaddle的方式，编译过程中`cmake`未设置`WITH_MKLDNN`的方式，
编译自定义OP时会报错找不到`mkldnn.h`等文件，可在`make.sh`中删除编译命令中的`-DPADDLE_WITH_MKLDNN`选项。

## 设置环境变量

需要将Paddle的核心库设置到`LD_LIBRARY_PATH`里, 先运行下面程序获取路径:

```
import paddle
print(paddle.sysconfig.get_lib())
```

可通过如下方式添加动态库路径:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python -c 'import paddle; print(paddle.sysconfig.get_lib())'`
```


更多关于如何在框架外部自定义 C++ OP，可阅读[官网说明文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_usage/index_cn.html)
