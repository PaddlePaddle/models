## 使用说明

`tf2paddle.py`提供了将TensorFlow训练的模型转换为PaddlePaddle可使用的模型的接口`TFModelConverter`，其封装了图像领域常用的Convolution、BatchNorm等layer的转换函数，可以完成VGG、ResNet等常用模型的转换。模型转换的基本过程是：基于TensorFlow的Python API获取variable，将各variable对应到PaddlePaddle中layer的参数，进行适配后序列化保存输出可以直接为PaddlePaddle的Python API加载使用的模型文件。

为使TensorFlow模型中的variable能够正确对应到PaddlePaddle模型中layer的参数，正确完成转换，模型转换具有如下约束：

- 支持TensorFlow中conv2d，batchnorm，fc这三种带有trainable variable的Operator中参数的转换。- TensorFlow配置中同一Operator内的variable属于相同的scope，以此将variable划分到不同的layer。
- conv2d、batchnorm、fc的scope需分别包含conv、bn、fc，以此获取对应layer的type；亦可以通过为`TFModelConverter`传入`layer_type_map`的`dict`，将scope映射到对应的layer type来规避此项约束。
- conv2d、fc中variable的顺序为先weight后bias，batchnorm中variable的顺序为scale、shift、mean、var，以此将variable对应到layer中相应位置的参数。
- TensorFlow网络拓扑顺序需和PaddlePaddle网络拓扑顺序一致，尤其注意具有分支时左右分支的顺序。这是针对模型转换和PaddlePaddle网络配置均使用PaddlePaddle默认参数命名的情况，此时将根据拓扑顺序进行参数命名；若PaddlePaddle网络配置中自定义了param的name，可以通过为`TFModelConverter`传入`layer_name_map`或`param_name_map`的`dict`，在模型转换时将variable的name映射为PaddlePaddle配置中param的name。

此外，要求提供`build_model`接口以从此构建TensorFlow网络，加载模型并返回session。可参照如下示例：

```python
def build_model():
    build_graph()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.tables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, 'model/model.ckpt')
    return sess
```

在完成以上内容后，`TFModelConverter`使用如下：

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

`tf2paddle.py`中已提供以上步骤，修改其中相关变量的值后执行`python tf2paddle.py`即可完成模型转换。

此外，在使用转换得到的模型时需要注意：

- 由于TensorFlow中的padding机制较为特殊，在编写PaddlePaddle网络配置时对conv这种需要padding的layer可能需要推算size后在conv外使用pad_layer进行padding。- 与TensorFlow多使用NHWC的data_format不同，PaddlePaddle使用NCHW的输入数据。
