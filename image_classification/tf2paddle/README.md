## 使用说明

`tf2paddle.py`脚本中的工具类`TFModelConverter`实现了将TensorFlow训练好的模型文件转换为PaddlePaddle可加载的模型文件。目前能够支持图像领域常用的：卷积（`Convolution`）层、`Batch Normalization`层和全连接（`Full Connection`）层。图像领域常用的 `ResNet` `VGG` 网络都以这些层此为基础，使用TensorFlow训练的`ResNet`和`VGG`模型能够被转换为PaddlePaddle可加载的模型，进一步用于预训练或是预测服务的开发等。

模型转换的基本流程是：
1. 将TensorFlow模型等价地使用PaddlePaddle Python API接口进行改写。
1. 在TensorFlow中可学习参数用 `Variable` 表示，基于TensorFlow的Python API获取网络中的 Variable。
1. 确定TensorFlow模型中`Variable`与PaddlePaddle中`paddle.layer`的可学习参数的对应关系。
1. 对TensorFlow中的`Variable`进行一定的适配（详见下文），转化为PaddlePaddle中的参数存储格式并进行序列化保存。

### 需要遵守的约定

为使TensorFlow模型中的`Variable`能够正确对应到`paddle.layer`中的可学习参数，目前版本在使用时有如下约束需要遵守：

1. 目前仅支持将TensorFlow中 `conv2d`，`batchnorm`，`fc`这三种带有可学习`Variable`的Operator训练出的参数向PaddlePaddle模型参数转换。
1. TensorFlow网络配置中同一Operator内的`Variable`属于相同的scope，以此为依据将`Variable`划分到不同的`paddle.layer`。
1. `conv2d`、`batchnorm`、`fc`的scope需分别包含`conv`、`bn`、`fc`，以此获取对应`paddle.layer`的类型。也可以通过为`TFModelConverter`传入`layer_type_map`的`dict`，将scope映射到对应的`paddle.layer`的type来规避此项约束。
1. `conv2d`、`fc`中`Variable`的顺序为：先可学习`Weight`后`Bias`；`batchnorm`中`Variable`的顺序为：`scale`、`shift`、`mean`、`var`，请注意参数存储的顺序将`Variable`对应到`paddle.layer.batch_norm`相应位置的参数。
1. TensorFlow网络拓扑顺序需和PaddlePaddle网络拓扑顺序一致，尤其注意网络包含分支结构时分支定义的先后顺序，如ResNet的bottleneck模块中两分支定义的先后顺序。这是针对模型转换和PaddlePaddle网络配置均使用PaddlePaddle默认参数命名的情况，此时将根据拓扑顺序进行参数命名。
1. 若PaddlePaddle网络配置中需要通过调用`param_attr=paddle.attr.Param(name="XX"))`显示地设置可学习参数名字，这时可通过为`TFModelConverter`传入`layer_name_map`或`param_name_map`字典（类型为Python `dict`），在模型转换时将`Variable`的名字映射为所对应的`paddle.layer.XX`中可学习参数的名字。
1. 要求提供`build_model`接口以从此构建TensorFlow网络，加载模型并返回session。可参照如下示例进行编写：

    ```python
    def build_model():
        build_graph()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(tf.tables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, 'model/model.ckpt')
        return sess
    ```

### 使用说明

按照以上原则操作后，`tf2paddle.py` 脚本的`main`函数提供了一个调用示例，将TensorFlow训练的`ResNet50`模型转换为PaddlePaddle可加载模型。若要对其它各种自定义的模型进行转换，只需修改相关变量的值，在终端执行`python tf2paddle.py`即可。

下面是一个简单的调用示例：

```python
# 定义相关变量
tf_net = "TF_ResNet50"                       # 提供build_model的module名
paddle_tar_name = "Paddle_ResNet50.tar.gz"   # 输出的Paddle模型的文件名

# 初始化并加载模型
converter = TFModelConverter(tf_net=tf_net,
                             paddle_tar_name=paddle_tar_name)
# 进行模型转换
converter.convert()
```

### 注意事项

1. 由于TensorFlow中的padding机制较为特殊，在编写PaddlePaddle网络配置时，对`paddle.layer.conv`这种需要padding的层可能需要推算size后在`paddle.layer.conv`外使用`paddle.layer.pad`进行padding。
1. 与TensorFlow图像输入多使用NHWC的数据组织格式有所不同，PaddlePaddle按照NCHW的格式组织图像输入数据。
