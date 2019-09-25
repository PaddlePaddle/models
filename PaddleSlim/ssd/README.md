本示例压缩目标为[MobileNetV1-SSD](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/ssd). 主要裁剪主干网络的卷积通道数。

## 第一步：观察网络结构

该模型的主干网络为MobileNetV1, 主要包含两种卷积：depthwise convolution和普通1X1卷积, 考虑到depthwise convolution的特殊性，我们只对普通1X1卷积做裁剪。

首先，我们需要知道主干网络中普通1X1卷积的参数（filter weights）的名称，在当前实现中，网络结构定义在`fluid.default_main_program`中，可以通过以下方式打印出网络中所有参数的名称和形状：

```
for param in fluid.default_main_program().global_block().all_parameters():
    print("{}: {}".format(param.name, param.shape))
```

上述代码会按网络定义顺序，依次打印相应的参数名称，如下所示：

```
conv2d_0.w_0 (32L, 3L, 3L, 3L)
depthwise_conv2d_0.w_0 (32L, 1L, 3L, 3L)
conv2d_1.w_0 (64L, 32L, 1L, 1L)
depthwise_conv2d_1.w_0 (64L, 1L, 3L, 3L)
conv2d_2.w_0 (128L, 64L, 1L, 1L)
depthwise_conv2d_2.w_0 (128L, 1L, 3L, 3L)
conv2d_3.w_0 (128L, 128L, 1L, 1L)
depthwise_conv2d_3.w_0 (128L, 1L, 3L, 3L)
conv2d_4.w_0 (256L, 128L, 1L, 1L)
depthwise_conv2d_4.w_0 (256L, 1L, 3L, 3L)
conv2d_5.w_0 (256L, 256L, 1L, 1L)
depthwise_conv2d_5.w_0 (256L, 1L, 3L, 3L)
conv2d_6.w_0 (512L, 256L, 1L, 1L)
depthwise_conv2d_6.w_0 (512L, 1L, 3L, 3L)
conv2d_7.w_0 (512L, 512L, 1L, 1L)
depthwise_conv2d_7.w_0 (512L, 1L, 3L, 3L)
conv2d_8.w_0 (512L, 512L, 1L, 1L)
depthwise_conv2d_8.w_0 (512L, 1L, 3L, 3L)
conv2d_9.w_0 (512L, 512L, 1L, 1L)
depthwise_conv2d_9.w_0 (512L, 1L, 3L, 3L)
conv2d_10.w_0 (512L, 512L, 1L, 1L)
depthwise_conv2d_10.w_0 (512L, 1L, 3L, 3L)
conv2d_11.w_0 (512L, 512L, 1L, 1L)
depthwise_conv2d_11.w_0 (512L, 1L, 3L, 3L)
conv2d_12.w_0 (1024L, 512L, 1L, 1L)
depthwise_conv2d_12.w_0 (1024L, 1L, 3L, 3L)

```

观察可知，普通1X1卷积名称为`conv2d_1.w_0`~`conv2d_12.w_0`, 用正则表达式可表示为：

```
"conv2d_[1-9].w_0|conv2d_1[0-2].w_0"
```

## 第二步：编写配置文件

我们以uniform pruning为例, 需要重点注意以下配置：

- target_ratio：指将被剪裁掉的flops的比例, 该选项的设置还要考虑主干网络参数量占全网络比例，如果该选项设置的太大，某些卷积层的channel会被全部裁剪掉，为了避免该问题，建议多先尝试设置不同的值，观察卷积层被裁剪的情况，然后再设置合适的值。当前示例会以0.2为例。

- pruned_params：将被裁剪的参数的名称，支持正则表达式，注意设置的正则表达式一定不要匹配到不想被裁剪到的参数名。最安全的做法是设置为`param_1_name|param_2_name|param_3_name`类似的格式，这样可以严格匹配指定的参数名。根据第一步，当前示例设置为`conv2d_[1-9].w_0|conv2d_1[0-2].w_0`


## 第三步：编写压缩脚本

当前示例的压缩脚本是在脚本[ssd/train.py](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/ssd/train.py)基础上修改的。

需要注意一下几点：

### fluid.metrics.DetectionMAP

 PaddleSlim暂时不支持fluid.metrics和fluid.evaluator, 所以这里将metrics.DetectionMAP改写为：

```
gt_label = fluid.layers.cast(x=gt_label, dtype=gt_box.dtype)
if difficult:
    difficult = fluid.layers.cast(x=difficult, dtype=gt_box.dtype)
    gt_label = fluid.layers.reshape(gt_label, [-1, 1])
    difficult = fluid.layers.reshape(difficult, [-1, 1])
    label = fluid.layers.concat([gt_label, difficult, gt_box], axis=1)
else:
    label = fluid.layers.concat([gt_label, gt_box], axis=1)

map_var = fluid.layers.detection.detection_map(
        nmsed_out,
        label,
        class_num,
        background_label=0,
        overlap_threshold=0.5,
        evaluate_difficult=False,
        ap_version=ap_version)
```

### data reader

注意在构造Compressor时，train_reader和eval_reader给的都是py_reader.
因为用了py_reader所以不需要再给feed_list.

```
   compressor = Compressor(
        place,
        fluid.global_scope(),
        train_prog,
        train_reader=train_py_reader, # noteeeeeeeeeeeee
        train_feed_list=None, # noteeeeeeeeeeeee
        train_fetch_list=train_fetch_list,
        eval_program=test_prog,
        eval_reader=test_py_reader, # noteeeeeeeeeeeee
        eval_feed_list=None, # noteeeeeeeeeeeee
        eval_fetch_list=val_fetch_list,
        train_optimizer=None)
```

## 第四步：保存剪裁模型

以下代码为保存剪枝后的模型:

```
com_pass = Compressor(...)
com_pass.config(args.compress_config)
com_pass.run()

pruned_prog = com_pass.eval_graph.program

fluid.io.save_inference_model("./pruned_model/", [image.name, label.name], [acc_top1], exe, main_program=pruned_prog)

# check the shape of parameters
for param in pruned_prog.global_block().all_parameters():
    print("name: {}; shape: {}".format(param.name, param.shape))
```

关于save_inference_model api请参考：https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/api_cn/io_cn.html#save-inference-model
