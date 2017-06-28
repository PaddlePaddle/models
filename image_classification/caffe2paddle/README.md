## 使用说明

`caffe2paddle.py`提供了将Caffe训练的模型转换为PaddlePaddle可使用的模型的接口`ModelConverter`，其封装了图像领域常用的Convolution、BatchNorm等layer的转换函数，可以完成VGG、ResNet等常用模型的转换。模型转换的基本过程是：基于Caffe的Python API加载模型并依次获取每一个layer的信息，将其中的参数根据layer类型与PaddlePaddle适配后序列化保存（对于Pooling等无需训练的layer不做处理），输出可以直接为PaddlePaddle的Python API加载使用的模型文件。

可以按如下方法使用`ModelConverter`接口：

```python
# 定义以下变量为相应的文件路径和文件名
caffe_model_file = "./ResNet-50-deploy.prototxt"        # Caffe网络配置文件的路径
caffe_pretrained_file = "./ResNet-50-model.caffemodel"  # Caffe模型文件的路径
paddle_tar_name = "Paddle_ResNet50.tar.gz"              # 输出的Paddle模型的文件名

# 初始化，从指定文件加载模型
converter = ModelConverter(caffe_model_file=caffe_model_file,
                           caffe_pretrained_file=caffe_pretrained_file,
                           paddle_tar_name=paddle_tar_name)
# 进行模型转换
converter.convert()
```

`caffe2paddle.py`中已提供以上步骤，修改其中文件相关变量的值后执行`python caffe2paddle.py`即可完成模型转换。此外，为辅助验证转换结果，`ModelConverter`中封装了使用Caffe API预测的接口`caffe_predict`，使用如下所示，将会打印按类别概率排序的(类别id, 概率)的列表:

```python
# img为图片路径，mean_file为图像均值文件的路径
converter.caffe_predict(img="./cat.jpg", mean_file="./imagenet/ilsvrc_2012_mean.npy")
```

需要注意，在模型转换时会对layer的参数进行命名，这里默认使用PaddlePaddle中默认的layer和参数命名规则：以`wrap_name_default`中的值和该layer类型的调用计数构造layer name，并以此为前缀构造参数名，比如第一个InnerProduct层（相应转换函数说明见下方）的bias参数将被命名为`___fc_layer_0__.wbias`。

```python
# 对InnerProduct层的参数进行转换，使用name值构造对应layer的参数名
# wrap_name_default设置默认name值为fc_layer
@wrap_name_default("fc_layer")
def convert_InnerProduct_layer(self, params, name=None)
```

为此，在验证和使用转换得到的模型时，编写PaddlePaddle网络配置无需指定layer name并且要保证和Caffe端模型使用同样的拓扑顺序，尤其是对于ResNet这种有分支的网络结构，要保证两分支在PaddlePaddle和Caffe中先后顺序一致，这样才能够使得模型参数正确加载。

如果不希望使用默认的命名，并且在PaddlePaddle网络配置中指定了layer name，可以建立Caffe和PaddlePaddle网络配置间layer name对应关系的`dict`并在调用`ModelConverter.convert`时作为`name_map`的值传入，这样在命名保存layer中的参数时将使用相应的layer name，不受拓扑顺序的影响。另外这里只针对Caffe网络配置中Convolution、InnerProduct和BatchNorm类别的layer建立`name_map`即可（一方面，对于Pooling等无需训练的layer不需要保存，故这里没有提供转换接口；另一方面，对于Caffe中的Scale类别的layer，由于Caffe和PaddlePaddle在实现上的一些差别，PaddlePaddle中的batch_norm层是BatchNorm和Scale层的复合，故这里对Scale进行了特殊处理）。
