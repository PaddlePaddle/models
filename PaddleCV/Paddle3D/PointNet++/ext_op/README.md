- 编译
```
cd src
sh make.sh
```
该编译会产出`pointnet_lib.so`

- 设置环境变量

需要将Paddle的核心库设置到`LD_LIBRARY_PATH`里, 先运行下面程序获取路径:
```
import paddle
print(paddle.sysconfig.get_lib())
```

假如路径为: `/paddle/pyenv/local/lib/python2.7/site-packages/paddle/libs`, 按如下方式设置即可。

```
export LD_LIBRARY_PATH=/paddle/pyenv/local/lib/python2.7/site-packages/paddle/libs:${LD_LIBRARY_PATH}
```

- 说明:

  - src: 扩展op C++/CUDA 源码
  - pointnet_lib.py: Python封装
  - tests: 单测程序
