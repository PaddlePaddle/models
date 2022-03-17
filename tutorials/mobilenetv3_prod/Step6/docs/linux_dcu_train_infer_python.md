## MobileNetV3


在Linux DCU 上进行训练、评估、预测等操作与Linux GPU基础使用教程一致，可参考[首页说明文档](../README.md)。启动命令与Linux GPU相同，只需在启动前设置环境变量HIP_VISIBLE_DEVICES(与CUDA_VISIBLE_DEVICES作用相同)，如下所示。

```shell
export HIP_VISIBLE_DEVICES=0
```
