# 自定义op的编译过程

## 代码结构

  - src: 扩展op C++/CUDA 源码
  - pointnet_lib.py: Python封装
  - tests: 单测程序

## 编译

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

- 如果您通过源码编译安装高于1.6版本的Paddle：

  动态库的编译命令被写入make.sh，可以直接执行编译脚本：

  ```
  cd src
  sh make.sh
  ```

- 如果您直接通过pip安装已发布的 paddlepaddle>=1.6.0 的 whl 包：

1. 首先您需要在本机下载 mkldnn：

```
cd ext_op
# 下载 mkldnn
git clone https://github.com/intel/mkl-dnn.git
# 切换到指定版本
cd mkl-dnn && git checkout aef88b7c233f48f8b945da310f1b973da31ad033
```

2. 然后需要将src/make.sh做如下修改，增加mkldnn的编译选项：

```
include_dir=$( python -c 'import paddle; print(paddle.sysconfig.get_include())' )
lib_dir=$( python -c 'import paddle; print(paddle.sysconfig.get_lib())' )

echo $include_dir
echo $lib_dir

OPS='farthest_point_sampling_op gather_point_op group_points_op query_ball_op three_interp_op three_nn_op'
for op in ${OPS}
do
nvcc ${op}.cu -c -o ${op}.cu.o -ccbin cc -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -DPADDLE_WITH_MKLDNN -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O0 -g -D_GLIBCXX_USE_CXX11_ABI=0 -DNVCC \
    -I ${include_dir}/third_party/install \
    -I ${include_dir}/third_party/install/gflags/include \
    -I ${include_dir}/third_party/install/glog/include \
    -I ${include_dir}/third_party/install/protobuf/include \
    -I ${include_dir}/third_party/install/xxhash/include \
    -I ${include_dir}/third_party/boost \
    -I ${include_dir}/third_party/eigen3 \
    -I ${include_dir}/third_party/threadpool/src/extern_threadpool \
    -I ${include_dir}/third_party/dlpack/include \
    -I ${include_dir}/third_party/ \
    -I ../mkl-dnn/include \ # 需替换成本地 mklnn 路径
    -I ${include_dir}
done

g++ farthest_point_sampling_op.cc farthest_point_sampling_op.cu.o gather_point_op.cc gather_point_op.cu.o group_points_op.cc group_points_op.cu.o query_ball_op.cu.o query_ball_op.cc three_interp_op.cu.o three_interp_op.cc three_nn_op.cu.o three_nn_op.cc -o pointnet_lib.so -DPADDLE_WITH_MKLDNN -shared -fPIC -std=c++11 -O0 -g -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I ${include_dir}/third_party/install/protobuf/include \
  -I ${include_dir}/third_party/install/glog/include \
  -I ${include_dir}/third_party/install/gflags/include \
  -I ${include_dir}/third_party/install/xxhash/include \
  -I ${include_dir}/third_party/install/zlib/include \
  -I ${include_dir}/third_party/boost \
  -I ${include_dir}/third_party/eigen3 \
  -I ${include_dir}/third_party/dlpack/include \
  -I ${include_dir}/third_party/ \
  -I ../mkl-dnn/include \ # 需替换成本地 mklnn 路径
  -I ${include_dir} \
  -L ${lib_dir} \
  -L /usr/local/cuda/lib64 -lpaddle_framework -lcudart

rm *.cu.o
```
为避免产生兼容问题，请使用**gcc4.8.2**版本编译custom-op

执行编译脚本：
```
cd ext_op/src
sh make.sh
```

最终编译会产出`pointnet_lib.so`

## 设置环境变量

需要将Paddle的核心库设置到`LD_LIBRARY_PATH`里, 先运行下面程序获取路径:

```
import paddle
print(paddle.sysconfig.get_lib())
```

假如路径为: `/paddle/pyenv/local/lib/python2.7/site-packages/paddle/libs`, 按如下方式设置即可。

```
export LD_LIBRARY_PATH=/paddle/pyenv/local/lib/python2.7/site-packages/paddle/libs:${LD_LIBRARY_PATH}
```

## 执行单测

执行下列单测，确保自定义算子可在网络中正确使用：

```
# 回到 ext_op 目录，添加 PYTHONPATH
cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`

# 运行单测 
python test/test_farthest_point_sampling_op.py
python test/test_gather_point_op.py
python test/test_group_points_op.py
python test/test_query_ball_op.py
python test/test_three_interp_op.py
python test/test_three_nn_op.py
```

单测运行成功会输出提示信息，如下所示：

```
.
----------------------------------------------------------------------
Ran 1 test in 13.205s

OK
```

更多关于如何在框架外部自定义 C++ OP，可阅读[官网说明文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_usage/index_cn.html)