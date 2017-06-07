## 使用说明

`caffe2paddle.py`提供了将Caffe训练的模型转换为PaddlePaddle可使用的模型的接口`ModelConverter`，其封装了图像领域常用的Convolution、BatchNorm等layer的转换函数，可完成VGG、ResNet等常用模型的转换。模型转换的基本过程是：基于Caffe的Python API加载模型并依次获取每一个layer的信息，将其中的参数根据layer类型与PaddlePaddle适配后序列化保存（对于Pooling等无需训练的layer不做处理），输出可以直接为PaddlePaddle的Python API加载使用的模型文件。

`ModelConverter`的定义及说明如下：

```python
class ModelConverter(object):
	#设置Caffe网络配置文件、模型文件路径和要保存为的Paddle模型的文件名，并使用Caffe API加载模型
    def __init__(self, caffe_model_file, caffe_pretrained_file, paddle_tar_name)

	#输出保存Paddle模型
    def to_tar(self, f)

	#将参数值序列化输出为二进制
    @staticmethod
    def serialize(data, f)

    #依次对各个layer进行转换，转换时参照name_map进行layer和参数命名
    def convert(self, name_map={})

	#对Caffe模型的Convolution层的参数进行转换，将使用name值对Paddle模型中对应layer的参数命名
    @wrap_name_default("img_conv_layer")
    def convert_Convolution_layer(self, params, name=None)

	#对Caffe模型的InnerProduct层的参数进行转换，将使用name值对Paddle模型中对应layer的参数命名
    @wrap_name_default("fc_layer")
    def convert_InnerProduct_layer(self, params, name=None)

	#对Caffe模型的BatchNorm层的参数进行转换，将使用name值对Paddle模型中对应layer的参数命名
    @wrap_name_default("batch_norm_layer")
    def convert_BatchNorm_layer(self, params, name=None)

	#对Caffe模型的Scale层的参数进行转换，将使用name值对Paddle模型中对应layer的参数命名
    def convert_Scale_layer(self, params, name=None)

	#输入图片路径和均值文件路径，使用加载的Caffe模型进行预测
    def caffe_predict(self, img, mean_file)

```

`ModelConverter`的使用方法如下：

```python
	#指定Caffe网络配置文件、模型文件路径和要保存为的Paddle模型的文件名，并从指定文件加载模型
    converter = ModelConverter("./ResNet-50-deploy.prototxt",
                               "./ResNet-50-model.caffemodel",
                               "Paddle_ResNet50.tar.gz")
    #进行模型转换
    converter.convert(name_map={})
    #进行预测并输出预测概率以便对比验证模型转换结果
    converter.caffe_predict(img='./caffe/examples/images/cat.jpg')
```

为验证并使用转换得到的模型，需基于PaddlePaddle API编写对应的网络结构配置文件，具体可参照PaddlePaddle使用文档，我们这里附上ResNet的配置以供使用。需要注意，上文给出的模型转换在调用`ModelConverter.convert`时传入了空的`name_map`，这将在遍历每一个layer进行参数保存时使用PaddlePaddle默认的layer和参数命名规则：以`wrap_name_default`中的值和调用计数构造layer name，并以此为前缀构造参数名（比如第一个InnerProduct层的bias参数将被命名为`___fc_layer_0__.wbias`）；为此，在编写PaddlePaddle网络配置时要保证和Caffe端模型使用同样的拓扑顺序，尤其是对于ResNet这种有分支的网络结构，要保证两分支在PaddlePaddle和Caffe中先后顺序一致，这样才能够使得模型参数正确加载。如果不希望使用默认的layer name，可以使用一种更为精细的方法：建立Caffe和PaddlePaddle网络配置间layer name对应关系的`dict`并在调用`ModelConverter.convert`时作为`name_map`传入，这样在命名保存layer中的参数时将使用相应的layer name，另外这里只针对Caffe网络配置中Convolution、InnerProduct和BatchNorm类别的layer建立`name_map`即可（一方面，对于Pooling等无需训练的layer不需要保存，故这里没有提供转换接口；另一方面，对于Caffe中的Scale类别的layer，由于Caffe和PaddlePaddle在实现上的一些差别，PaddlePaddle中的batch_norm层同时包含BatchNorm和Scale层的复合，故这里对Scale进行了特殊处理）。
