**层叠降噪自编码器**
---

**概述**
---
层叠降噪自编码器（Stacked Denoising Autoencoders，SDA）[1]通过将降噪自编码器叠加而成，将前一层降噪自编码器的输出作为当前层的输入，这种预训练方式每次只初始化一层参数，其中每一层降噪自编码器通过最小化重构误差进行训练。当所有层都训练结束后，进行第二阶段的监督训练。降噪自编码器（Denoising Autoencoders，DA）是在自编码器的基础上，通过在训练数据中加入噪声，以学习到更加鲁邦的特征表示，模型泛化能力更强。

**原理**
---
SDA的基础是自编码器，自编码器包含编码器和解码器，其中编码器将输入向量$\textbf{x}$转化为隐层表示：
$$\textbf{y}=s(\textbf{Wx}+\textbf{b})$$
其中$\textbf{W}$是权重矩阵，$\textbf{b}$是偏置。解码器将隐层表示映射为输入重建向量：
$$\textbf{z}=s(\textbf{W}'\textbf{y}+\textbf{b}')$$
自编码器的训练采用交叉熵损失训练：
$$L(\textbf{x},\textbf{z})=-\sum_{k=1}^{d}[\textbf{x}_klog\textbf{z}_k+(1-\textbf{x}_k)log(1-\textbf{z}_k)]$$

**示例总览**
---

本示例共包含以下文件：

表1. 示例文件

 文件                              | 用途                                    |
-------------------------         | -------------------------------------   |
 autoencoder.py    | 自编码器定义脚本                      |  
 train_da.py       | 自编码器训练脚本      |  
 stacked_autoencoder.py| 层叠降噪自编码器定义脚本            |  
 train_sda.py     | 层叠降噪自编码器训练脚本        |  
 utils.py                          | 常用函数脚本    |  
 infer.py          | 测试脚本                  |  
 train_sda.sh      | SDA训练shell脚本  |


---

**实验**
---
**降噪自编码器**

本部分通过可视化的方式研究降噪自编码器学习到的特征，实验中采用MNIST数据集，使用交叉熵损失训练自编码器，采用zero-masking噪声。在训练数据中加入不同比例的噪声，观察自编码器学习到特征的不同。
直接执行`python train_da.py`即可训练自编码器，同时将学习到的特征可视化到当前文件夹。其中自编码器(autoencoder.py)定义：
```python
def denoise_autoencoder(input, args):
    n_hidden = args.n_hidden
    n_visible = args.img_height * args.img_width
    W = fluid.layers.create_parameter(shape=[n_visible, n_hidden], dtype='float32', attr=fluid.ParamAttr(name='W', initializer=fluid.initializer.Normal()), is_bias=False)
    bvis = fluid.layers.zeros(shape=[n_visible], dtype='float32')
    bhid = fluid.layers.zeros(shape=[n_hidden], dtype='float32')
    hidden_value = fluid.layers.sigmoid(fluid.layers.matmul(input, W) + bhid)
    out = fluid.layers.sigmoid(fluid.layers.matmul(hidden_value, W, transpose_y=True) + bvis)
    return out
```
实验结果如下所示，在未添加噪声的情况下，一部分滤波器未激活，随着噪声比例的增加，学习到的滤波器更加具有判别性。实验中添加的噪声比例为：0，25%和50%。
![DA学习的参数可视化结果](https://github.com/chengyuz/models/blob/yucheng_sda/fluid/sda/images/da_res.png)

**层叠降噪自编码器**

本部分比较层叠降噪自编码器（SDA）和普通层叠自编码器的分类性能，其中SDA采用降噪自编码器进行逐层预训练初始化，普通层叠自编码器使用高斯分布进行初始化。实验采用MNIST数据集，SDA采用zero-masking噪声。
其中层叠自编码器（stacked_autoencoder.py）定义：
```python
def build_model(self, layer_input):
    for i in range(len(self.num_layers)):
        sigmoid_layer = fluid.layers.fc(
            input=layer_input,
            size=self.num_layers[i],
            act='sigmoid',
            param_attr=fluid.ParamAttr(
                name='s%d_w' % i, initializer=fluid.initializer.Normal()),
            bias_attr=fluid.ParamAttr(
                name='s%d_b' % i, initializer=fluid.initializer.Constant()))
        layer_input = sigmoid_layer

    out = fluid.layers.fc(
        input=sigmoid_layer,
        size=self.class_num,
        act='softmax',
        param_attr=fluid.ParamAttr(
            name='out_w', initializer=fluid.initializer.Normal()),
        bias_attr=fluid.ParamAttr(
            name='out_b', initializer=fluid.initializer.Constant()))
    return out
```
实验中逐层训练自编码器作为深度模型的初始化参数，具体来说：
1. 运行`train_sda.sh`训练SDA，训练好的模型保存在`models/SDAE`文件夹
2. 运行`python train_sda.py --mode sda --pretrain_strategy SAE`训练普通层叠自编码器，训练好的模型保存在`models/SAE`文件夹
3. 运行`python infer.py --mode SDAE`测试SDA模型
4. 运行`python infer.py --mode SAE`测试普通层叠自编码器

实验结果如下：

 Model                  | Top1 Accuracy                   |
 -------------------------         | -------------------------------------   |
 SDA    |        0.967               |
普通层叠自编码器|  0.961  |

可以看出SDA相比普通层叠自编码器识别精度更高，表明了使用降噪自编码器预训练的有效性。

**引用**
---
1. Jonathan Long, Evan Shelhamer, Trevor Darrell. [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](http://www.jmlr.org/papers/v11/vincent10a.html), JMLR2010.
